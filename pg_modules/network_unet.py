import torch
import torch.nn as nn
import torch.nn.functional as F
import pg_modules.layers as layers
import functools
from torch.nn import init

def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}

    arch[256] = {'in_channels' :    [ch * item for item in [16, 16, 8, 8, 4, 2]],
                             'out_channels' : [ch * item for item in [16, 8, 8, 4, 2, 1]],
                             'upsample' : [True] * 6,
                             'resolution' : [8, 16, 32, 64, 128, 256],
                             'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                                                            for i in range(3,9)}}
    arch[128] = {'in_channels' :    [ch * item for item in [16, 16, 8, 4, 2]],
                             'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
                             'upsample' : [True] * 5,
                             'resolution' : [8, 16, 32, 64, 128],
                             'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                                                            for i in range(3,8)}}
    return arch

class Generator(nn.Module):
    def __init__(self, ngf=64, z_dim=128, bottom_width=4, img_resolution=256,
                             G_kernel_size=3, G_attn='64', n_classes=0,
                             num_G_SVs=1, num_G_SV_itrs=1,
                             G_shared=False, shared_dim=0, hier=True,
                             cross_replica=False, mybn=False,
                             G_activation=nn.ReLU(inplace=False),
                             BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
                             G_init='ortho', skip_init=True, no_optim=False,
                             G_param='SN', norm_style='bn',
                             **kwargs):
        super(Generator, self).__init__()
        # Channel width mulitplier
        self.ch = ngf
        # Dimensionality of the latent space
        self.z_dim = z_dim
        self.c_dim = 0
        self.w_dim = 0
        self.img_channels = 3
        # The initial spatial dimensions
        self.bottom_width = bottom_width
        # Resolution of the output
        self.img_resolution = img_resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # number of classes, for use in categorical conditional generation
        self.n_classes = n_classes
        # Use shared embeddings?
        self.G_shared = G_shared
        # Dimensionality of the shared embedding? Unused if not using G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else z_dim
        # Hierarchical latent space?
        self.hier = hier
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # fp16?
        self.fp16 = G_fp16
        # Architecture dict
        self.arch = G_arch(self.ch, self.attention)[img_resolution]

        self.unconditional = True

        # If using hierarchical latents, adjust z
        if self.hier:
            # Number of places z slots into
            self.num_slots = len(self.arch['in_channels']) + 1
            self.z_chunk_size = (self.z_dim // self.num_slots)

            if not self.unconditional:
                self.z_dim = self.z_chunk_size *    self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                    kernel_size=3, padding=1,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear

        # We use a non-spectral-normed embedding here regardless;
        # For some reason applying SN to G's embedding seems to randomly cripple G
        self.which_embedding = nn.Embedding

        if self.unconditional:
            bn_linear = nn.Linear
            input_size =  self.z_dim  + (self.shared_dim if self.G_shared else 0 )
        else:
            bn_linear = (functools.partial(self.which_linear, bias = False) if self.G_shared
                                     else self.which_embedding)

            input_size = (self.shared_dim + self.z_chunk_size if self.G_shared
                                    else self.n_classes)
        self.which_bn = functools.partial(layers.ccbn,
                                                    which_linear=bn_linear,
                                                    cross_replica=self.cross_replica,
                                                    mybn=self.mybn,
                                                    input_size=input_size,
                                                    norm_style=self.norm_style,
                                                    eps=self.BN_eps,
                                                    self_modulation = self.unconditional)


        # Prepare model
        # If not using shared embeddings, self.shared is just a passthrough
        self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared
                                        else layers.identity())
        # First linear layer
        if self.unconditional:
            self.linear = self.which_linear(self.z_dim, self.arch['in_channels'][0] * (self.bottom_width **2))
        else:
            self.linear = self.which_linear(self.z_dim // self.num_slots,
                                                                        self.arch['in_channels'][0] * (self.bottom_width **2))

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):


            self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                                                     out_channels=self.arch['out_channels'][index],
                                                     which_conv=self.which_conv,
						                             which_bn=self.which_bn,
                                                     activation=self.activation,
                                                     upsample=(functools.partial(F.interpolate, scale_factor=2)
                                                                         if self.arch['upsample'][index] else None))]]

            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                                                                cross_replica=self.cross_replica,
                                                                                                mybn=self.mybn),
                                                                        self.activation,
                                                                        self.which_conv(self.arch['out_channels'][-1], 3))

        # Initialize weights. Optionally skip init for testing.
        if not skip_init:
            self.init_weights()

        # Set up optimizer
        # If this is an EMA copy, no need for an optim, so just return now
        # if no_optim:
        #     return
        # self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
        # if G_mixed_precision:
        #     print('Using fp16 adam in G...')
        #     import utils
        #     self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
        #                                              betas=(self.B1, self.B2), weight_decay=0,
        #                                              eps=self.adam_eps)
        # else:
        #     self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
        #                                              betas=(self.B1, self.B2), weight_decay=0,
        #                                              eps=self.adam_eps)

        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)

    # Note on this forward function: we pass in a y vector which has
    # already been passed through G.shared to enable easy class-wise
    # interpolation later. If we passed in the one-hot and then ran it through
    # G.shared in this forward function, it would be harder to handle.
    def forward(self, z, c, noise_mode='const'):
        # If hierarchical, concatenate zs and ys
        if self.hier:
            # faces
            if self.unconditional:
                ys = [z for _ in range(self.num_slots)]
            else:
                zs = torch.split(z, self.z_chunk_size, 1)
                z = zs[0]

                ys = [torch.cat([c, item], 1) for item in zs[1:]]
        else:
            if self.unconditional:
                ys = [None] * len(self.blocks)
            else:
                ys = [c] * len(self.blocks)

        # First linear layer
        h = self.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                # print('h ', h.shape)
                # print('ys[index] ', ys[index].shape)
                h = block(h, ys[index])

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))

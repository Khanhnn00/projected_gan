from pg_modules.discriminator import ProjectedDiscriminator, MixDiscriminator
import torch
import dnnlib

# model = Unet_Discriminator()

input = torch.randn((16, 3, 256, 256))
# output = model(input)
# print(output.shape)

backbone_kwargs = dnnlib.EasyDict()

backbone_kwargs.cout = 64
backbone_kwargs.expand = True
backbone_kwargs.proj_type = 2
backbone_kwargs.num_discs = 4
backbone_kwargs.separable = False
backbone_kwargs.cond = False

# pj_model = ProjectedDiscriminator(backbone_kwargs=backbone_kwargs)

mix_model = MixDiscriminator(backbone_kwargs=backbone_kwargs,D_mixed_precision=True)
out_pj = mix_model.proj(input, label='real')
out_unet = mix_model.unet(input)

print(out_pj.shape)
print(out_unet.shape)



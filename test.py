import torch
import torch.nn as nn
import torch.nn.functional as F
from pg_modules.blocks import (DownBlock, InitLayer, UpBlockBig, UpBlockBigCond, UpBlockSmall, UpBlockSmallCond, SEBlock, conv2d, GLU, Swish)

class Conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.02, inplace=True)
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

 
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Conv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = Conv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = Conv(in_channels, out_channels*2)
            self.norm = nn.BatchNorm2d(out_channels*2)
            self.act = GLU()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        # return x
        x = self.norm(x)
        return self.act(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            Swish(),
            conv2d(out_channels, out_channels, 1, 1, 0, bias=True),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = Conv(self.in_channels, 64)
        self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        # self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.out_channels)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x = self.up1(x3, x2)
        x = self.up2(x2, x1)
        logits = self.outc(x)
        return logits


import dnnlib
import legacy
device = 'cuda:2'

# ckp_object = '/home/ubuntu/runs/00011-fastgan_lite-mix-ada-cityscape_train_256-gpus4-batch144-fpn0-unet0.200000/network-snapshot.pkl'
# with dnnlib.util.open_url(ckp_object, verbose=True) as f:
#     network_dict = legacy.load_network_pkl(f)
#     G_main = network_dict['G_ema'].eval().to(device) # subclass of torch.nn.Module
    

# unet = UNet(3, 3).to(device)
# inp = torch.ones((16,3,256,256)).to(device)
# res = unet(inp)
# print(res.shape)
# print(res.max(), res.min(), res.mean())

# z_main = torch.randn([16, 256], device=device)
# res = G_main(z_main, c=0)
# print(res.max(), res.min(), res.mean())
# print(torch.cuda.max_memory_allocated(device)/ 2**30)

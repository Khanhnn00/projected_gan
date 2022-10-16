from pg_modules.discriminator import ProjectedDiscriminator, MixDiscriminator, UNetDiscriminator
import torch
import sys
sys.path.insert(0, './yolov7')
import dnnlib
from yolov7.detector import predict_img
import cv2
import os
import numpy as np
import torchvision.transforms as transforms
import dnnlib
import legacy
from pg_modules.hungarian import HungarianMatcher

device = 'cuda'
resize = transforms.Resize(size=(640, 640))
bz = 8

ckp_object = '/home/ubuntu/runs/00011-fastgan_lite-mix-ada-cityscape_train_256-gpus4-batch144-fpn0-unet0.200000/network-snapshot.pkl'
with dnnlib.util.open_url(ckp_object, verbose=True) as f:
    network_dict = legacy.load_network_pkl(f)
    G_main = network_dict['G_ema'] # subclass of torch.nn.Module

ckp_object = '/home/ubuntu/run_unet/00022-fastgan_lite-mix-ada-crop_train_256-gpus4-batch144/network-snapshot.pkl'
with dnnlib.util.open_url(ckp_object, verbose=True) as f:
    network_dict = legacy.load_network_pkl(f)
    G_unet = network_dict['G_ema'] # subclass of torch.nn.Module
    
G_main = G_main.eval().to(device)
G_unet = G_unet.eval().to(device)

k= 20
nsamples = k*bz
lo, hi = -1, 1

z_main = torch.randn([bz, G_unet.z_dim], device=device)
z_unet = torch.randn([nsamples, G_unet.z_dim], device=device)
samples = G_main(z_main, c=0).float()
new_samples = (samples - lo) * (255 / (hi - lo))
new_samples = torch.clip(new_samples.round(), min=0, max=255)

objects = G_unet(z_unet, c=0)
outputs = predict_img(new_samples)
new_objects = (objects - lo) * (255 / (hi - lo))
new_objects = torch.clip(new_objects.round(), min=0, max=255)

matcher = HungarianMatcher()

idx = 0
print(len(outputs))
print('**********************')
for i in range(len(outputs)):
    print(len(outputs[i]))
    samples[i, :, :, :] = matcher(new_samples[i], outputs[i], new_objects[idx:idx+k], samples[i], objects[idx:idx+k])[0]
    idx += k
    
for i in range(bz):
    tmp = samples[i].detach().cpu().numpy()
    tmp = np.transpose(tmp, (1,2,0))[:, :, ::-1]
    tmp = np.ascontiguousarray(tmp)
    cv2.imwrite('{}.jpg'.format(i), tmp)
    
    



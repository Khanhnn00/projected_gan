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
# model = UNetDiscriminator().to(device)

# input = torch.randn((2, 3, 256, 256))
# output = model(input)
# print(output.shape, output.max(), output.min(), output.mean())

# backbone_kwargs = dnnlib.EasyDict()

# backbone_kwargs.cout = 64
# backbone_kwargs.expand = True
# backbone_kwargs.proj_type = 2
# backbone_kwargs.num_discs = 4
# backbone_kwargs.separable = False
# backbone_kwargs.cond = False

# pj_model = ProjectedDiscriminator(backbone_kwargs=backbone_kwargs)

# mix_model = MixDiscriminator(backbone_kwargs=backbone_kwargs,D_mixed_precision=True)
# out_pj = mix_model.proj(input, label='real')
# out_unet = mix_model.unet(input)

# print(out_pj.shape)
# print(out_unet.shape)

path = '/home/ubuntu/test_cityscape'
img_list = os.listdir(path)
batch = []
for i in img_list:
    img = cv2.imread(os.path.join(path, i))[:, :, ::-1]
    img = torch.tensor(np.transpose(np.ascontiguousarray(img), (2,0,1))).unsqueeze(0)
    batch.append(img)

batch = torch.cat(batch, dim=0).to(device).float()
outputs = predict_img(batch)
# print(outputs[0])
# print(res, len(res), len(res[0]))

ckp_object = '/home/ubuntu/run_unet/00022-fastgan_lite-mix-ada-crop_train_256-gpus4-batch144/network-snapshot.pkl'
with dnnlib.util.open_url(ckp_object, verbose=True) as f:
    network_dict = legacy.load_network_pkl(f)
    G_unet = network_dict['G_ema'] # subclass of torch.nn.Module
    
G_unet = G_unet.eval().to(device)

k= 20
nsamples = k*len(outputs)

z_unet = torch.randn([nsamples, G_unet.z_dim], device=device)
objects = G_unet(z_unet, c=0)
lo, hi = -1, 1
objects = np.asarray(objects.cpu(), dtype=np.float32)
objects = (objects - lo) * (255 / (hi - lo))
objects = np.rint(objects).clip(0, 255).astype(np.uint8)
objects = torch.from_numpy(objects).to(device).float()

matcher = HungarianMatcher()

idx = 0
print(len(outputs))
print('**********************')
for i in range(len(outputs)):
    print(len(outputs[i]))
    batch[i, :, :, :] = matcher(batch[i], outputs[i], objects[idx:idx+k])
    idx += k
    
for i in range(len(img_list)):
    tmp = batch[i].detach().cpu().numpy()
    tmp = np.transpose(tmp, (1,2,0))[:, :, ::-1]
    tmp = np.ascontiguousarray(tmp)
    cv2.imwrite(img_list[i], tmp)
    
    



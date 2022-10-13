import torch
import os
import random
import torchvision.transforms as transforms
resize = transforms.Resize(size=(640, 640))
import cv2
import numpy as np

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
half = device != 'cpu'
save_img = True
save_txt = True

model = attempt_load('yolov7/yolov7.pt', map_location=device)
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
stride = int(model.stride.max())  # model stride
# imgsz = check_img_size(640, s=stride)  # check img_size
# model.eval()
# model.requires_grad_(False)
model.eval()
model = model.to(device)
iou_thres=0.6
conf_thres = 0.5

if not os.path.exists('images_uploaded'):
    os.mkdir('images_uploaded')
    
if not os.path.exists('video_saved'):
    os.mkdir('video_saved')
    
save_dir = 'results_detect'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def predict_img(im0s): #B 3 H W   
    
    imgs = im0s.clone()
    h0, w0 = imgs.shape[2::]  # orig hw
    # print(h0, w0)
    r = 640 / max(h0, w0)  # resize image to img_size
    # print(max(h0, w0), r*h0, r*w0)
    if r != 1:  # always resize down, only resize up if training with augmentation
        imgs = transforms.Resize(size=(int(h0 * r), int(w0 * r)))(imgs)
    res = []
    cnt = 0
    for img, im0 in zip(imgs, im0s):
        im0 = im0.cpu().numpy()
        k_bbox = []
        img /= 255.
        img = img.unsqueeze(0)
        # print(img.shape)
        im0 = np.transpose(im0, (1,2,0))
        
        out = model(img)[0]
        
        # print(out.shape)
        pred = non_max_suppression(out, conf_thres=0.25, iou_thres=iou_thres, multi_label=True)
        for i, det in enumerate(pred):  # detections per image
            # print(det, det.size())
            
            # gn = torch.tensor(im0s[k].shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det = det.detach()
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                
                for *xyxy, conf, cls in reversed(det):
                    tlbr = torch.tensor(xyxy).tolist()
                    sub = im0[int(tlbr[1]):int(tlbr[3]), int(tlbr[0]):int(tlbr[2]), ::-1]
                    sub = np.ascontiguousarray(sub)
                    # print(sub.shape[0] * sub.shape[1])
                    if sub.shape[0] * sub.shape[1] >= 0.5*128*128:
                        cnt += 1
                        cv2.imwrite('res/{}.jpg'.format(cnt), sub)
                        k_bbox.append([tlbr, cls])
        res.append(k_bbox)
    
    return res
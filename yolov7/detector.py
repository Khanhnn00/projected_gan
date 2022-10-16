import torch
import os
import random
import torchvision.transforms as transforms
resize = transforms.Resize(size=(640, 640))
import cv2
import numpy as np

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
half = device != 'cpu'
save_img = True
save_txt = True

iou_thres=0.6
conf_thres = 0.5

def predict_img(model, im0s): #B 3 H W   
    imgs = im0s.clone()
    h0, w0 = imgs.shape[2::]  # orig hw
    r = 640 / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        imgs = transforms.Resize(size=(int(h0 * r), int(w0 * r)))(imgs)
    res = []
    cnt = 0
    for img, im0 in zip(imgs, im0s):
        # im0 = im0.cpu().numpy()
        k_bbox = []
        img = img/255.
        img = img.unsqueeze(0)
        # print(img.shape)
        im0 = torch.permute(im0, (1,2,0))
        
        with torch.no_grad():
            out = model(img)[0]
        
        # print(out.shape)
            pred = non_max_suppression(out, conf_thres=0.25, iou_thres=iou_thres, multi_label=True)
        for i, det in enumerate(pred):  # detections per image
            # print(det, det.size())
            
            # gn = torch.tensor(im0s[k].shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                try:
                    det = det.detach()
                except Exception as e:
                    pass
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                
                for *xyxy, conf, cls in reversed(det):
                    tlbr = torch.tensor(xyxy).tolist()
                    sub = im0[int(tlbr[1]):int(tlbr[3]), int(tlbr[0]):int(tlbr[2]), :]
                    # sub = np.ascontiguousarray(sub)
                    # print(sub.shape[0] * sub.shape[1])
                    if sub.shape[0] * sub.shape[1] >= 0.5*32*32:
                        cnt += 1
                        k_bbox.append([tlbr, cls])
        res.append(k_bbox)
    
    return res
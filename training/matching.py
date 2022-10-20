from yolov7.detector import predict_img
import torch

def matching(model, matcher, samples, objects, k, half, g_device, g_unet_device):
    
    hi, lo = 1, -1
    new_samples = (samples - lo) * (255 / (hi - lo))
    new_samples = torch.clip(new_samples.round(), min=0, max=255)
    new_objects = (objects - lo) * (255 / (hi - lo))
    new_objects = torch.clip(new_objects.round(), min=0, max=255)
    outputs = predict_img(model, new_samples.half() if half else new_samples)
    idx = 0
    for i in range(len(outputs)):
        # print(len(outputs[i]))
        samples[i, :, :, :] = matcher(new_samples[i], new_objects[idx:idx+k], samples[i], objects[idx:idx+k], outputs[i], g_device, g_unet_device)[1]
        idx += k
        
    return samples
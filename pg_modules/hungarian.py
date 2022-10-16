# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn as nn
import numpy as np

loss = torch.nn.MSELoss()
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()


    @torch.no_grad()
    def forward(self, batch, targets, org_samples, org_targets, outputs, g_device, g_unet_device):
        """ Performs the matching

        Params:
            outputs: list
            targets: B 3 H W

        Returns:

        """
        # assert len(outputs) == targets.shape[0]

        # We flatten to compute the cost matrices in a batch
        subTarget = [np.transpose(targets[k].cpu().numpy(), (1,2,0)) for k in range(targets.shape[0])]
        subOrgTarget = [np.transpose(org_targets[k].cpu().numpy(), (1,2,0)) for k in range(org_targets.shape[0])]
        for ind, pred in enumerate(outputs):
            if len(subTarget) == 0:
                pass
            else:
                pred_bbox, cls = pred[0], pred[1]
                h, w = abs(pred_bbox[3] - pred_bbox[1]), abs(pred_bbox[2] - pred_bbox[0])
                tmp_transform = transforms.Resize(size=(int(h), int(w)))
                cur_idx = -1
                curLoss = np.iinfo(np.uint32).max
                curPatch = None
                for k in range(len(subTarget)):
                    # print(subTarget[k].shape)
                    h, w = subTarget[k].shape[0], subTarget[k].shape[1]
                    # print(h, w)
                    mh, mw = h//2, w//2
                    th = bh = mh
                    lw = rw = mw
                    # print(subTarget[k][:, :, 0])
                    while subTarget[k][mh][rw][0] != 0 and rw < w-1:
                        rw += 1
                    while subTarget[k][mh][lw][0] != 0 and lw > 0:
                        lw -= 1
                    
                    while subTarget[k][th][mw][0] != 0 and th < h-1:
                        th += 1
                    while subTarget[k][bh][mw][0] != 0 and bh > 0:
                        bh -= 1
                    tmp_target = subTarget[k][bh:th+1, lw:rw+1, :]
                    tmp_org_target = subOrgTarget[k][bh:th+1, lw:rw+1, :]
                    # print(tmp_target.shape)
                    tmp_target = tmp_transform(torch.from_numpy(tmp_target).to(g_device).permute(2,0,1))
                    tmp_org_target = tmp_transform(torch.from_numpy(tmp_org_target).to(g_device).permute(2,0,1))
                    tmp = loss(batch[:, int(pred_bbox[1]):int(pred_bbox[3]), \
                                int(pred_bbox[0]):int(pred_bbox[2])],\
                                tmp_target)
                    # print(tmp)
                    if tmp.item() < curLoss:
                        curLoss = tmp
                        curPatch = tmp_target
                        curOrg = tmp_org_target
                        cur_idx = k
                subTarget.pop(cur_idx)
                batch[:, int(pred_bbox[1]):int(pred_bbox[3]), \
                                int(pred_bbox[0]):int(pred_bbox[2])] = curPatch
                org_samples[:, int(pred_bbox[1]):int(pred_bbox[3]), \
                                int(pred_bbox[0]):int(pred_bbox[2])] = curOrg
                
        return batch, org_samples
                
def build_matcher(args):
    return HungarianMatcher()
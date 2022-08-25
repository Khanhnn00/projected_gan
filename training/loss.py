# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
#
import sys
import os
import inspect
import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(1, currentdir)

from pg_modules.fpn import FPN101

import random
import lpips
# percept = lpips.LPIPS(net='vgg')


class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()


class ProjectedGANLoss(Loss):
    def __init__(self, device, G, D, G_ema, blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.fpn = FPN101().to(device).requires_grad_(False)
        self.l2_loss = torch.nn.MSELoss()

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        img = self.G.synthesis(ws, c, update_emas=False)
        return img

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']: return  # no regularization needed for PG

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0

        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Gmain = (-gen_logits).mean()

                real_fts = self.fpn(real_img)
                fake_fts = self.fpn(gen_img)
                loss_ft = 0
                for (real_ft, fake_ft) in zip(real_fts, fake_fts):
                    loss_ft += self.l2_loss(real_ft, fake_ft).mean()
                
                loss_Gmain += loss_ft

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Dgen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_img_tmp = real_img.detach().requires_grad_(False)
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                loss_Dreal = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

            with torch.autograd.profiler.record_function('Dreal_backward'):
                loss_Dreal.backward()

class FastGANLoss(Loss):
    def __init__(self, device, G, G_ema, D):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.percept = percept.to(device)

    def crop_image_by_part(self, image, part):
        hw = image.shape[2]//2
        if part==0:
            return image[:,:,:hw,:hw]
        if part==1:
            return image[:,:,:hw,hw:]
        if part==2:
            return image[:,:,hw:,:hw]
        if part==3:
            return image[:,:,hw:,hw:]

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        img = self.G.synthesis(ws, c, update_emas=False)
        return img

    def run_D(self, net, img, label="real"):
        if label=="real":
            part = random.randint(0, 3)
            pred, [rec_all, rec_small, rec_part] = net(img, label, part=part)
            err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
                percept( rec_all, F.interpolate(img, rec_all.shape[2]) ).sum() +\
                percept( rec_small, F.interpolate(img, rec_small.shape[2]) ).sum() +\
                percept( rec_part, F.interpolate(crop_image_by_part(img, part), rec_part.shape[2]) ).sum()
            err.backward()
            return pred.mean().item(), rec_all, rec_small, rec_part
        else:
            pred = net(img, label)
            err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
            err.backward()
            return pred.mean().item()

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']: return  # no regularization needed for PG

        if do_Dmain:
            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                gen_img = self.run_G(gen_z, gen_c)
                part = random.randint(0, 3)
                pred, [rec_all, rec_small, rec_part] = self.D(real_img, 'real', part=part)
                D_real = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
                    self.percept( rec_all, F.interpolate(gen_img, rec_all.shape[2]) ).sum() +\
                    self.percept( rec_small, F.interpolate(gen_img, rec_small.shape[2]) ).sum() +\
                    self.percept( rec_part, F.interpolate(self.crop_image_by_part(gen_img, part), rec_part.shape[2]) ).sum()
                
                dreal_pred = pred.mean().item()

            with torch.autograd.profiler.record_function('Dreal_backward'):
                D_real.backward()
                # err_dr, rec_img_all, rec_img_small, rec_img_part = self.runD(self.D, gen_img, label="real")
            with torch.autograd.profiler.record_function('Dgen_forward'):
                # real_img_tmp = real_img.detach().requires_grad_(False)
                # # real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                # loss_Dreal = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                # # Logging
                # training_stats.report('Loss/scores/real', real_logits)
                # training_stats.report('Loss/signs/real', real_logits.sign())
                # training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                
                gen_img_tmp = gen_img.detach().requires_grad_(False)
                pred = self.D(gen_img_tmp, 'fake')
                D_fake = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()

                # D_fake = 0
                # for fi in gen_img:
                #     pred = self.D(fi.detach().unsqueeze(0), 'fake')
                #     err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
                #     D_fake += err
                # self.runD(self.D, [fi.detach() for fi in gen_img], label="fake")

                # Logging
                training_stats.report('Loss/scores/real', dreal_pred)
                training_stats.report('Loss/signs/real', D_real.sign())
                training_stats.report('Loss/D/loss', D_fake + D_real)

            with torch.autograd.profiler.record_function('Dgen_backward'):
                D_fake.backward()

            # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(gen_z, gen_c)
                pred_g = self.D(gen_img, "fake")
                err_g = -pred_g.mean()

                # Logging
                training_stats.report('Loss/scores/fake', pred_g.mean())
                training_stats.report('Loss/signs/fake', pred_g.mean().sign())
                training_stats.report('Loss/G/loss', err_g)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                err_g.backward()
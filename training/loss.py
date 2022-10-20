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
from imp import is_frozen_package
import random
from cmath import pi
import functools
import sys
import os
import inspect
import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
import dnnlib

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
sys.path.insert(1, currentdir)


# percept = lpips.LPIPS(net='vgg')
# for param in percept.parameters():
#     param.requires_grad = False


class Loss:
    # to be overridden by subclass
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        raise NotImplementedError()


class ProjectedGANLoss(Loss):
    def __init__(self, device, G, D, G_ema, blur_init_sigma=0, blur_fade_kimg=0, is_fpn=False, pixel_wise='L2', is_percept=False, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.is_fpn = is_fpn
        self.pixel_wise = pixel_wise
        self.is_percept = is_percept
        self.pixLoss = None

        if self.is_fpn and self.pixel_wise == None:
            raise Exception("There must be pixel-wise loss to use FPN")
        
        if pixel_wise == 'L1':
            self.pixLoss = torch.nn.L1Loss()
        elif pixel_wise == 'L2':
            self.pixLoss = torch.nn.MSELoss()

        if is_fpn:
            from pg_modules.fpn import FPN101
            self.fpn = FPN101().to(device).requires_grad_(False)

        if is_percept:
            import lpips
            self.percept = lpips.LPIPS(net='vgg').to(device)
            for param in self.percept.parameters():
                param.requires_grad = False

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        img = self.G.synthesis(ws, c, update_emas=False)
        return img

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1,
                                 device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']:
            return  # no regularization needed for PG

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * \
            self.blur_init_sigma if self.blur_fade_kimg > 1 else 0

        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Gmain = (-gen_logits).mean()

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
                loss_Dgen = (F.relu(torch.ones_like(
                    gen_logits) + gen_logits)).mean()

                if self.is_fpn:

                    real_fts = self.fpn(real_img)
                    fake_fts = self.fpn(gen_img)
                    loss_ft = 0
                    for (real_ft, fake_ft) in zip(real_fts, fake_fts):
                        loss_ft += self.l1_loss(real_ft, fake_ft).mean()

                    loss_Dgen += loss_ft

                if self.is_percept:
                    percelt_loss = self.percept(real_img, gen_img).mean()
                    loss_Dgen += percelt_loss

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_img_tmp = real_img.detach().requires_grad_(False)
                real_logits = self.run_D(
                    real_img_tmp, real_c, blur_sigma=blur_sigma)
                loss_Dreal = (F.relu(torch.ones_like(
                    real_logits) - real_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

            with torch.autograd.profiler.record_function('Dreal_backward'):
                loss_Dreal.backward()


class UNetLoss(Loss):
    def __init__(self, G, G_ema, D, device) -> None:
        self.G = G
        self.G_ema = G_ema
        self.device = device
        self.unet = D
        # self.requires_grad(True)

        # self.d_real_target = torch.tensor([1.0]).to(self.device)
        # self.d_fake_target = torch.tensor([0.0]).to(self.device)

    def requires_grad(self, flag=True):
        for p in self.D.parameters():
            p.requires_grad = flag

    def run_G(self, z, c, update_emas=False):
        # ws = self.G.mapping(z, c, update_emas=update_emas)
        # img = self.G.synthesis(ws, c, update_emas=False)
        img = self.G(z, c)
        return img

    def run_D(self, img, flag=True, update_emas=False):
        prob_map = self.unet(img) #1 256 256
        return prob_map.view(-1)

    def train(self, G, D, G_optin, D_optim, gen_z, gen_c):
        D.zero_grad()

        gen_img = self.run_G(gen_z, gen_c, update_emas=True)
        gen_logits = self.run_D(gen_img, False)
        loss_Dgen = (F.relu(torch.ones_like(
            gen_logits) + gen_logits)).mean()

        # Logging
        training_stats.report('Loss/scores/fake', gen_logits)
        training_stats.report('Loss/signs/fake', gen_logits.sign())

        # real_img_tmp = real_img.detach().requires_grad_(False)
        real_logits = self.run_D(gen_img.detach(), False)
        loss_Dreal = (F.relu(torch.ones_like(
            real_logits) - real_logits)).mean()

        # Logging
        training_stats.report('Loss/scores/real', real_logits)
        training_stats.report('Loss/signs/real', real_logits.sign())
        training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)


    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']:
            return  # no regularization needed for PG

        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, True)
                loss_Gmain = (-gen_logits).mean()

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('D_forwad'):
                gen_img = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img)
                loss_Dgen = (F.relu(torch.ones_like(
                    gen_logits) + gen_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            # with torch.autograd.profiler.record_function('Dgen_backward'):
            #     loss_Dgen.backward()

            # Dmain: Maximize logits for real images.
            # with torch.autograd.profiler.record_function('Dreal_forward'):
                # gen_img = self.run_G(gen_z, gen_c)
                real_img_tmp = real_img.detach().requires_grad_(False)
                real_logits = self.run_D(real_img_tmp)
                loss_Dreal = (F.relu(torch.ones_like(
                    real_logits) - real_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

            # with torch.autograd.profiler.record_function('Dreal_backward'):
            #     loss_Dreal.backward()
            with torch.autograd.profiler.record_function('D_backward'):
                total = loss_Dreal+loss_Dgen
                total.backward()

class MixLoss(Loss):
    def __init__(self, device, G, D, G_ema, blur_init_sigma=0, blur_fade_kimg=0, is_fpn=False, \
                 pixel_wise='L1', is_percept=False, weight_unet=1e-2, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.is_fpn = is_fpn
        self.pixel_wise = pixel_wise
        self.is_percept = is_percept
        self.weight_unet = weight_unet
        # print(is_percept, is_fpn, weight_unet)

        if self.is_fpn and self.pixel_wise == None:
            raise Exception("There must be pixel-wise loss to use FPN")
        
        if pixel_wise == 'L1':
            self.pixLoss = torch.nn.L1Loss()
        elif pixel_wise == 'L2':
            self.pixLoss = torch.nn.MSELoss()

        if is_fpn:
            from pg_modules.fpn import FPN101
            self.fpn = FPN101().to(device).requires_grad_(False)

        if is_percept:
            import lpips
            self.percept = lpips.LPIPS(net='vgg').to(device)
            for param in self.percept.parameters():
                param.requires_grad = False

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        img = self.G.synthesis(ws, c, update_emas=False)
        return img

    def run_D_proj(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1,
                                 device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        logits = self.D.proj(img, c)
        return logits

    def run_D_unet(self, img):
        prob_map = self.D.unet(img) #1 256 256
        return prob_map.view(-1)

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']:
            return  # no regularization needed for PG

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * \
            self.blur_init_sigma if self.blur_fade_kimg > 1 else 0

        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D_proj(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Gmain = (-gen_logits).mean()

                # Logging
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D_proj(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Dgen = (F.relu(torch.ones_like(
                    gen_logits) + gen_logits)).mean()

                gen_logits_unet = self.run_D_unet(gen_img)
                loss_Dgen_unet = (F.relu(torch.ones_like(
                    gen_logits_unet) + gen_logits_unet)).mean()

                if self.is_fpn:

                    real_fts = self.fpn(real_img)
                    fake_fts = self.fpn(gen_img)
                    loss_ft = 0
                    for (real_ft, fake_ft) in zip(real_fts, fake_fts):
                        loss_ft += self.pixLoss(real_ft, fake_ft).mean()

                    loss_Dgen += loss_ft

                if self.is_percept:
                    percelt_loss = self.percept(real_img, gen_img).mean()
                    loss_Dgen += percelt_loss

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                training_stats.report('Loss/scores/fake/unet', gen_logits_unet)
                training_stats.report('Loss/signs/fake/unet', gen_logits_unet.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                total_dgen = loss_Dgen + self.weight_unet*loss_Dgen_unet
                total_dgen.backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_img_tmp = real_img.detach().requires_grad_(False)
                real_logits = self.run_D_proj(
                    real_img_tmp, real_c, blur_sigma=blur_sigma)
                loss_Dreal = (F.relu(torch.ones_like(
                    real_logits) - real_logits)).mean()

                real_logits_unet = self.run_D_unet(gen_img.detach())
                loss_Dreal_unet = (F.relu(torch.ones_like(
                    real_logits_unet) - real_logits_unet)).mean()

                if self.is_fpn:

                    real_fts = self.fpn(real_img)
                    fake_fts = self.fpn(gen_img)
                    loss_ft = 0
                    for (real_ft, fake_ft) in zip(real_fts, fake_fts):
                        loss_ft += self.pixLoss(real_ft, fake_ft).mean()

                    loss_Dreal += loss_ft

                if self.is_percept:
                    percelt_loss = self.percept(real_img, gen_img).mean()
                    loss_Dreal += percelt_loss

                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                training_stats.report('Loss/scores/real/unet', real_logits_unet)
                training_stats.report('Loss/signs/real/unet', real_logits_unet.sign())

                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal + loss_Dgen_unet + loss_Dreal_unet)

            with torch.autograd.profiler.record_function('Dreal_backward'):
                total_dreal = loss_Dreal + self.weight_unet*loss_Dreal_unet
                total_dreal.backward()
                
class ProjectedHungarianLoss(Loss):
    def __init__(self, device, G, D, G_ema, blur_init_sigma=0, blur_fade_kimg=0, is_fpn=False, \
                 pixel_wise='L1', is_percept=False, **kwargs) -> None:
        super().__init__()
        import sys
        sys.path.insert(0, './yolov7')
        from yolov7.models.experimental import attempt_load
        self.device = device
        self.G = G
        ckp_object = '/home/ubuntu/run_unet/00022-fastgan_lite-mix-ada-crop_train_256-gpus4-batch144/network-snapshot.pkl'
        import legacy
        with dnnlib.util.open_url(ckp_object, verbose=True) as f:
            network_dict = legacy.load_network_pkl(f)
            self.G_unet = network_dict['G_ema'].requires_grad_(False).to(device) # subclass of torch.nn.Module
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.is_fpn = is_fpn
        self.pixel_wise = pixel_wise
        self.is_percept = is_percept
        from training.matching import matching as matchingfn
        from pg_modules.hungarian import HungarianMatcher
        self.matcher = HungarianMatcher()
        self.matchingfn = matchingfn
        self.k = 10
        self.detector = attempt_load('yolov7/yolov7.pt', map_location=device)
        self.detector.eval()
        self.half = device != 'cpu'
        if self.half:
            self.detector.half()
       
    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        img = self.G.synthesis(ws, c, update_emas=False)
        z_unet = torch.randn([self.k*img.shape[0], self.G_unet.z_dim], device=self.device)
        
        with torch.no_grad():
            objects = self.G_unet(z_unet, c=0)
        
        img = self.matchingfn(self.detector, self.matcher, img, objects, self.k, self.half,\
                              next(self.G.parameters()).device, next(self.G_unet.parameters()).device)
        img = self.G.refine(img)
        return img
    
    def run_D(self, img, c, blur_sigma=0, update_emas=False):

        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1,
                                 device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        logits = self.D(img, c)
        return logits
        
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']:
            return  # no regularization needed for PG

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * \
            self.blur_init_sigma if self.blur_fade_kimg > 1 else 0

        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Gmain = (-gen_logits).mean()

                # Logging
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                loss_Dgen = (F.relu(torch.ones_like(
                    gen_logits) + gen_logits)).mean()

                if self.is_fpn:

                    real_fts = self.fpn(real_img)
                    fake_fts = self.fpn(gen_img)
                    loss_ft = 0
                    for (real_ft, fake_ft) in zip(real_fts, fake_fts):
                        loss_ft += self.pixLoss(real_ft, fake_ft).mean()

                    loss_Dgen += loss_ft

                if self.is_percept:
                    percelt_loss = self.percept(real_img, gen_img).mean()
                    loss_Dgen += percelt_loss

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())


            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_img_tmp = real_img.detach().requires_grad_(False)
                real_logits = self.run_D(
                    real_img_tmp, real_c, blur_sigma=blur_sigma)
                loss_Dreal = (F.relu(torch.ones_like(
                    real_logits) - real_logits)).mean()
                
                if self.is_fpn:

                    real_fts = self.fpn(real_img)
                    fake_fts = self.fpn(gen_img)
                    loss_ft = 0
                    for (real_ft, fake_ft) in zip(real_fts, fake_fts):
                        loss_ft += self.pixLoss(real_ft, fake_ft).mean()

                    loss_Dreal += loss_ft

                if self.is_percept:
                    percelt_loss = self.percept(real_img, gen_img).mean()
                    loss_Dreal += percelt_loss

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
        if part == 0:
            return image[:, :, :hw, :hw]
        if part == 1:
            return image[:, :, :hw, hw:]
        if part == 2:
            return image[:, :, hw:, :hw]
        if part == 3:
            return image[:, :, hw:, hw:]

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        img = self.G.synthesis(ws, c, update_emas=False)
        return img

    def run_D(self, net, img, label="real"):
        if label == "real":
            part = random.randint(0, 3)
            pred, [rec_all, rec_small, rec_part] = net(img, label, part=part)
            err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean() + \
                percept(rec_all, F.interpolate(img, rec_all.shape[2])).sum() +\
                percept(rec_small, F.interpolate(img, rec_small.shape[2])).sum() +\
                percept(rec_part, F.interpolate(
                    self.crop_image_by_part(img, part), rec_part.shape[2])).sum()
            err.backward()
            return pred.mean().item(), rec_all, rec_small, rec_part
        else:
            pred = net(img, label)
            err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
            err.backward()
            return pred.mean().item()

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']:
            return  # no regularization needed for PG

        if do_Dmain:
            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                gen_img = self.run_G(gen_z, gen_c)
                part = random.randint(0, 3)
                pred, [rec_all, rec_small, rec_part] = self.D(
                    real_img, 'real', part=part)
                D_real = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean() + \
                    self.percept(rec_all, F.interpolate(gen_img, rec_all.shape[2])).sum() +\
                    self.percept(rec_small, F.interpolate(gen_img, rec_small.shape[2])).sum() +\
                    self.percept(rec_part, F.interpolate(
                        self.crop_image_by_part(gen_img, part), rec_part.shape[2])).sum()

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
                D_fake = F.relu(torch.rand_like(pred) *
                                0.2 + 0.8 + pred).mean()

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

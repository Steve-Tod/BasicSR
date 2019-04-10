import os
import random
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss, GradientPenaltyLoss
from models.modules.sampler import random_sampler, adaptive_sampler, pyramid_sampler
logger = logging.getLogger('base')



class SRRaGANMultiModel(BaseModel):
    def __init__(self, opt):
        super(SRRaGANMultiModel, self).__init__(opt)
        train_opt = opt['train']
        
        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netD = networks.define_D(opt)  # D
            for k in self.netD.keys():
                self.netD[k] = self.netD[k].to(self.device)
            self.netG.train()
            for k in self.netD.keys():
                self.netD[k].train()
        self.load()  # load G and D if needed
        self.num_D = train_opt['num_D']
        sampler_swith = {
                'random': random_sampler,
                'adaptive': adaptive_sampler,
                'pyramid': pyramid_sampler
                }
        self.sampler = sampler_swith[train_opt['sample_strategy']]
        if train_opt['sample_strategy'] == 'adaptive':
            if 'num_candidate' not in train_opt.keys():
                raise KeyError('Use adaptive sample strategy but no num_candidate in train_opt')
            self.num_candidate = train_opt['num_candidate']
        else:
            self.num_candidate = None

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters are for WGAN
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            if train_opt['gan_type'] == 'wgan-gp':
                self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
                # gradient penalty loss
                self.cri_gp = GradientPenaltyLoss(device=self.device).to(self.device)
                self.l_gp_w = train_opt['gp_weigth']

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else {'128': 0, '64': 0, '32': 0} 
            self.optimizer_D = {}
            for k in self.netD.keys():
                self.optimizer_D[k] = torch.optim.Adam(self.netD[k].parameters(), lr=train_opt['lr_D'][k], 
                weight_decay=wd_D[k], betas=(train_opt['beta1_D'][k], 0.999))
                self.optimizers.append(self.optimizer_D[k])

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        # print network
        self.print_network()

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L = data['LR'].to(self.device)

        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)

            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        # G
        for k in self.netD.keys():
            for p in self.netD[k].parameters():
                p.requires_grad = False

        self.optimizer_G.zero_grad()

        self.fake_H = self.netG(self.var_L)

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.var_H).detach()
                fake_fea = self.netF(self.fake_H)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea
            # G gan + cls loss
            # Multi D
            pic_size = (self.fake_H.shape[2], self.fake_H.shape[3])
            if self.num_candidate:
                with torch.no_grad():
                    dist_map = torch.sum(torch.abs(self.fake_H - self.var_ref), dim=(0, 1))
                    crop_res_dict = self.sampler(pic_size, self.num_D, self.num_candidate, dist_map)
            else:
                crop_res_dict = self.sampler(pic_size, self.num_D)

            l_g_gan_sum = 0.0
            for k in self.netD.keys():
                for c in crop_res_dict[k]:
                    start_y, end_y, start_x, end_x = c
                    pred_g_fake = self.netD[k](self.fake_H[:, :, start_y: end_y, start_x: end_x])
                    pred_d_real = self.netD[k](self.var_ref[:, :, start_y: end_y, start_x: end_x]).detach()

                    l_g_gan = self.l_gan_w[k] * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                          self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
                    l_g_total += l_g_gan
                    l_g_gan_sum += l_g_gan.item()

            l_g_total.backward()
            self.optimizer_G.step()

        # D
        for k in self.netD.keys():
            for p in self.netD[k].parameters():
                p.requires_grad = True

        for k in self.optimizer_D.keys():
            self.optimizer_D[k].zero_grad()

        l_d_total = 0

        # Multi D
        pic_size = (self.fake_H.shape[2], self.fake_H.shape[3])
        if self.num_candidate:
            crop_res_dict = random_sampler(pic_size, self.num_D)
        else:
            crop_res_dict = self.sampler(pic_size, self.num_D)

        l_d_real_sum = 0.0
        l_d_fake_sum = 0.0
        for k in self.netD.keys():
            for c in crop_res_dict[k]:
                start_y, end_y, start_x, end_x = c
                pred_d_real = self.netD[k](self.var_ref[:, :, start_y: end_y, start_x: end_x])
                pred_d_fake = self.netD[k](self.fake_H.detach()[:, :, start_y: end_y, start_x: end_x])  # detach to avoid BP to G

                l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
                l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
                l_d_total += (l_d_real + l_d_fake) / 2
                l_d_real_sum += l_d_real.item()
                l_d_fake_sum += l_d_fake.item()

        if self.opt['train']['gan_type'] == 'wgan-gp':
            pic_size = (self.fake_H.shape[2], self.fake_H.shape[3])
            crop_size = (int(k), int(k))
            batch_size = self.var_ref.size(0)
            if self.random_pt.size(0) != batch_size:
                self.random_pt.resize_(batch_size, 1, 1, 1)
            self.random_pt.uniform_()  # Draw random interpolation points
            interp = self.random_pt * self.fake_H.detach() + (1 - self.random_pt) * self.var_ref
            interp.requires_grad = True
            l_d_gp_sum = 0.0
            for k in self.netD.keys():
                for _ in range(self.num_D[k]):
                    start_y, end_y, start_x, end_x  = _get_random_crop_indices(pic_size, crop_size)
                    interp_crit, _ = self.netD[k](interp[:, :, start_y: end_y, start_x: end_x])
                    l_d_gp = self.l_gp_w * self.cri_gp(interp[:, :, start_y: end_y, start_x: end_x], interp_crit)  # maybe wrong in cls?
                    l_d_total += l_d_gp
                    l_d_gp_sum += l_d_gp.item()

        l_d_total.backward()
        for k in self.optimizer_D.keys():
            self.optimizer_D[k].step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # G
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['l_g_gan'] = l_g_gan_sum
        # D 
        self.log_dict['l_d_real'] = l_d_real_sum
        self.log_dict['l_d_fake'] = l_d_fake_sum

        if self.opt['train']['gan_type'] == 'wgan-gp':
            self.log_dict['l_d_gp'] = l_d_gp_sum
        # D outputs
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Networks G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)
        if self.is_train:
            # Discriminator
            for k, d in self.netD.items():
                s, n = self.get_network_description(d)
                if isinstance(d, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(d.__class__.__name__,
                                                    d.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(d.__class__.__name__)

                logger.info('Network D_{} structure: {}, with parameters: {:,d}'.format(k, net_struc_str, n))
                logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                    self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)

                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading pretrained model for D [{:s}] ...'.format(str(load_path_D)))
            if isinstance(load_path_D, str):
                for k in self.opt['train']['num_D'].keys():
                    p = load_path_D.split('.')[0] + k + '.pth'
                    self.load_network(p, self.netD[k])
            else:
                for k, p in load_path_D.items():
                    self.load_network(p, self.netD[k])

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        for k, d in self.netD.items():
            self.save_network(d, 'D' + k, iter_step)

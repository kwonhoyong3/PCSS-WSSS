from cProfile import label
from http.client import NON_AUTHORITATIVE_INFORMATION
import os, time
from statistics import mode
import os.path as osp
import argparse
import glob
import random
import pdb
from turtle import pos
import math

import numpy as np
from numpy.core.fromnumeric import size
# from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools

# Image tools
import cv2
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from torchvision import transforms
import torchvision

import voc12.data
from tools import utils, pyutils, trmutils
from tools.imutils import save_img, denorm, _crf_with_alpha, cam_on_image
# import tools.visualizer as visualizer
from networks import mctformer

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

import shutil

import sys
sys.path.append("..") 

def set_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class RepulsionLoss(torch.nn.Module):
    def __init__(self, strength=0.1, radius=2):
        super().__init__()
        self.strength = strength
        self.radius = radius

    def forward(self, x):
        differences = x.unsqueeze(-1) - x.unsqueeze(-2) #B C C C
        distances = differences.abs().sum(dim=1) # B C C
        repulsion_weights = (distances < self.radius).float() * self.strength
        repulsion_offsets = differences * repulsion_weights.unsqueeze(-1)
        loss = repulsion_offsets.sum(dim=-2).norm(p=2, dim=-1).mean()
        return loss

class model_WSSS():

    def __init__(self, args, logger):

        self.args = args

        if self.args.C == 20:
            self.categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                           'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person',
                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
            self.categories_withbg = ['bg','aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                           'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person',
                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        elif self.args.C == 80:
            self.categories= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
                        'bus', 'train', 'truck', 'boat', 'traffic_light',
                        'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird',
                        'cat', 'dog', 'horise', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
                        'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'bottle',
                        'wine_glass', 'cup', 'fork', 'knife', 'spoon',
                        'bowl', 'banana', 'apple', 'sandwich', 'orange',
                        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut',
                        'cake', 'chair', 'couch', 'potted_plant', 'bed',
                        'dining_table', 'toilet', 'tv', 'laptop', 'mouse',
                        'remote', 'keyboard', 'cell_phone', 'microwave', 'oven',
                        'toaster', 'sink', 'refrigerator', 'book', 'clock',
                        'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']
            
        # Common things
        self.phase = 'train'
        self.dev = 'cuda'
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_wosig = nn.BCELoss()
        self.cs = nn.CosineSimilarity(dim=0)
        self.bs = args.batch_size
        self.logger = logger
        self.writer = args.writer

        # Model attributes
        self.net_names = ['net_trm']
        self.base_names = ['cls', 'cls_hif_reduced', 'cam_hif_reduced', 'cls_merge', 'cam_merge']
        self.loss_names = ['loss_' + bn for bn in self.base_names]
        self.acc_names = ['acc_' + bn for bn in self.base_names]

        self.ebd_memory_t = []
        self.ebd_memory_s = []

        self.is_empty_memory = [True for i in range(len(self.categories))]



        self.nets = []
        self.opts = []
        
        # Evaluation-related
        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.accs = [0] * len(self.acc_names)
        self.count = 0
        self.num_count = 0
        
        #Tensorboard
        self.global_step = 0

        self.val_wrong = 0
        self.val_right = 0

        self.num_class = args.C

        # Define networks
        self.net_trm = create_model(
            'deit_small_MCTformerV2_PCSS_patch16_224',
            pretrained=False,
            num_classes=args.C,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None
        )

            
        if args.finetune:
           
            checkpoint = torch.load("./pretrained/deit_small_patch16_224-cd65a155.pth", map_location='cpu')

            try:
                checkpoint_model = checkpoint['model']
            except:
                checkpoint_model = checkpoint
            state_dict = self.net_trm.state_dict()

            if 'head.bias' in state_dict.keys():
                for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]
            else:
                for k in ['head.weight', 'head_dist.weight', 'head_dist.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]

            # interpolate position embedding
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = self.net_trm.patch_embed.num_patches
            if args.finetune.startswith('https'):
                num_extra_tokens = 1
            else:
                num_extra_tokens = self.net_trm.pos_embed.shape[-2] - num_patches

            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)

            new_size = int(num_patches ** 0.5)

            if args.finetune.startswith('https') and 'MCTformer' in args.trm:
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens].repeat(1,args.C,1)  
            else:
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]

            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

            if args.finetune.startswith('https') and 'MCTformer' in args.trm:
                cls_token_checkpoint = checkpoint_model['cls_token']
                perturb = torch.randn_like(cls_token_checkpoint.repeat(1,args.C,1))           
                sign = cls_token_checkpoint.repeat(1,args.C,1).sign()                        
                new_cls_token = cls_token_checkpoint.repeat(1,args.C,1)
                
                checkpoint_model['cls_token'] = new_cls_token             
            

            self.net_trm.load_state_dict(checkpoint_model, strict=False)

        self.L2 = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.KD = nn.KLDivLoss(reduction='batchmean')
        
        def gaussian_fn(M, std):
            n = torch.arange(0, M) - (M - 1.0) / 2.0
            sig2 = 2 * std * std
            w = torch.exp(-n ** 2 / sig2)
            return w

        def gkern(kernlen=256, std=128):
            """Returns a 2D Gaussian kernel array."""
            gkern1d = gaussian_fn(kernlen, std=std) 
            gkern2d = torch.outer(gkern1d, gkern1d)
            return gkern2d

       
        self.loss_mat_for_freq = {}

        self.loss_mat_for_freq_cur = torch.zeros((self.num_class,224,224)).to(self.dev)

        self.fim_ipc = args.fim_ipc    
        self.mpc_idx = args.mpc_idx             
        self.mpc_org_ratio = args.mpc_org_ratio    

    # Save networks
    def save_model(self, epo, ckpt_path):
        epo_str = str(epo).zfill(3)
        torch.save(self.net_trm.module.state_dict(), ckpt_path + '/' + epo_str + 'net_trm.pth')

    # Load networks
    def load_model(self, epo, ckpt_path):
        epo_str = str(epo).zfill(3)
        self.net_trm.load_state_dict(torch.load(ckpt_path + '/' + epo_str + 'net_trm.pth'), strict=True)

        self.net_trm = torch.nn.DataParallel(self.net_trm.to(self.dev))

    def load_pretrained_model(self, pretrained_path):
        self.net_trm.load_state_dict(torch.load(pretrained_path), strict=True)

        self.net_trm = torch.nn.DataParallel(self.net_trm.to(self.dev))
        
    # Set networks' phase (train/eval)
    def set_phase(self, phase):

        if phase == 'train':
            self.phase = 'train'
            for name in self.net_names:
                getattr(self, name).train()
            self.logger.info('Phase : train')

        elif phase == 'freq':
            self.phase = 'freq_aggregate'
            for name in self.net_names:
                getattr(self, name).eval()
            self.logger.info('Phase : freq_aggregate')

            self.fim_ipc_stack = torch.zeros((self.num_class)).to(self.dev) + self.fim_ipc

        else:
            self.phase = 'eval'
            for name in self.net_names:
                getattr(self, name).eval()
            self.logger.info('Phase : eval')

    # Set optimizers and upload networks on multi-gpu
    def train_setup(self):

        args = self.args


        linear_scaled_lr = args.lr * args.batch_size * trmutils.get_world_size() / 512.0
        args.lr = linear_scaled_lr

        self.opt_trm = create_optimizer(args, self.net_trm)
        self.lr_scheduler, _ = create_scheduler(args, self.opt_trm)


        self.logger.info('Poly-optimizer for trm is defined.')

        self.net_trm = torch.nn.DataParallel(self.net_trm.to(self.dev))
        self.logger.info('Networks are uploaded on multi-gpu.')

        self.nets.append(self.net_trm)

    # Unpack data pack from data_loader
    def unpack(self, pack, for_freq=False):

        if self.phase == 'train':
            self.img = pack[0].to(self.dev)  # B x 3 x H x W
            self.label = pack[1].to(self.dev)  # B x 20
            self.name = pack[2]  # list of image names

        if self.phase == 'freq_aggregate':
            self.img = pack[0].to(self.dev)  # B x 3 x H x W
            self.label = pack[1].to(self.dev)  # B x 20
            self.name = pack[2]  # list of image names

        if self.phase == 'eval':
            self.img = pack[0]
            self.label = pack[1].to(self.dev)
            self.name = pack[2][0]

    # Do forward/backward propagation and call optimizer to update the networks
    def update(self, epo):
        # Tensor dimensions
        B = self.img.shape[0]
        H = self.img.shape[2]
        W = self.img.shape[3]

        h,w = H//16, W//16
        C = self.num_class  # Number of cls

        self.B = B
        self.C = C

        ################################################### Update TRM ###################################################
        self.opt_trm.zero_grad()
        self.net_trm.train()

        loss_trm = torch.Tensor([0]).cuda()

        outputs = self.net_trm(self.img, self.mpc_idx)

        self.out = outputs['cls']
        self.out_patch = outputs['pcls']
        
        cams = outputs['cams']
        rcams = outputs['rcams']
        x_patch_list = outputs['x_patch_list']

        self.loss_cls = 1*(
            F.multilabel_soft_margin_loss(self.out,self.label)
            + F.multilabel_soft_margin_loss(self.out_patch,self.label) 
        )
        loss_trm += self.loss_cls 
        
        
        ################# Frequency Shortcut Suppression #################
        if self.args.W[0] > 0 and epo > self.args.fim_start_epo:

            # Make high Influence Frequency Map mask        
            hifm_mask = self.high_influence_freq_mask_cur  # 20 H W

            new_imgs = torch.zeros_like(self.img).to(self.dev)
            new_mask = torch.ones_like(hifm_mask).to(self.dev)
            new_mask = new_mask - hifm_mask
            
            valid_img_idx = torch.ones((B)).to(self.dev)
            for b in range(B):
                exist_label = torch.nonzero(self.label[b])[:,0]         # N
                label_num = exist_label.shape[0]
                if label_num == 0:
                    ''' Ignore no class image from loss '''
                    new_imgs[b] = self.img[b]
                    valid_img_idx[b] = 0
                    continue

                imgs = self.img[b]
                img_freq = torch.zeros(imgs.size(), dtype=torch.complex128).to(self.dev)
                img_freq = torch.fft.fftshift(torch.fft.fft2(imgs))
                
                imgs_freq_filtered = img_freq.view(1,3,H,W) * new_mask[exist_label].view(label_num,1,H,W)
                hif_reduced_img = torch.fft.ifft2(torch.fft.ifftshift(imgs_freq_filtered))
                hif_reduced_img = torch.real(hif_reduced_img)
                hif_reduced_img = torch.Tensor(hif_reduced_img)     # N 3 H W

                new_imgs[b] = hif_reduced_img.mean(dim=0)

            hif_reduced_outputs = self.net_trm(new_imgs, self.mpc_idx)      # B 3 H W in

            self.out_hif_reduced = hif_reduced_outputs['cls']
            self.out_patch_hif_reduced = hif_reduced_outputs['pcls']
            cams_hif_reduced = hif_reduced_outputs['cams']            # B 20 H W

            
            self.loss_cls_hif_reduced = 1*(
                F.multilabel_soft_margin_loss(self.out_hif_reduced,self.label)
                + F.multilabel_soft_margin_loss(self.out_patch_hif_reduced,self.label)
            )
            loss_trm += self.loss_cls_hif_reduced 


            self.loss_cam_hif_reduced = self.args.W[0]*(
                ((self.max_norm(cams) - self.max_norm(cams_hif_reduced)) * valid_img_idx.view(B,1,1,1)).abs().mean()
            )
            loss_trm += self.loss_cam_hif_reduced

        else:
            self.loss_cls_hif_reduced = torch.Tensor([0])
            self.loss_cam_hif_reduced = torch.Tensor([0])

        ################ Magnitude-mixing based Phase Concentration ###################
        if self.args.W[1] > 0:
            
            batch_idx_permuted = torch.randperm(B).cuda()
            
            x_patch = x_patch_list[0] # B D H W                     # Get only 1 patch of merge_block_idx inside network
            x_patch_for_merge = x_patch[batch_idx_permuted]
            
            mag1 = torch.abs(torch.fft.fftn(x_patch, dim=(2,3)))     # fft through HW
            mag2 = torch.abs(torch.fft.fftn(x_patch_for_merge, dim=(2,3)))

            mag_merged = mag1 * self.mpc_org_ratio + mag2.detach() * (1-self.mpc_org_ratio)

            outputs_mag_merge = self.net_trm(self.img, self.mpc_idx, mag_merged)
            out_merge = outputs_mag_merge['cls']
            out_patch_merge = outputs_mag_merge['pcls']
            cams_merge = outputs_mag_merge['cams']
            rcams_merge = outputs_mag_merge['rcams']

            self.loss_cls_merge = 1*(
                F.multilabel_soft_margin_loss(out_merge,self.label)
                + F.multilabel_soft_margin_loss(out_patch_merge,self.label)
            )
            loss_trm += self.loss_cls_merge
            
            if epo >= self.args.mpc_cam_start_epo:
                self.loss_cam_merge = self.args.W[1]*(
                    ((self.max_norm(rcams).detach() - self.max_norm(rcams_merge))).abs().mean()
                )
                loss_trm += self.loss_cam_merge
            else:
                self.loss_cam_merge = torch.Tensor([0])
            
        else:
            self.loss_cls_merge = torch.Tensor([0])
            self.loss_cam_merge = torch.Tensor([0])


        loss_trm.backward()
        self.opt_trm.step()
        
        ################################################### Export ###################################################


        for i in range(len(self.loss_names)):
            self.running_loss[i] += getattr(self, self.loss_names[i]).item()

        self.count += 1
        #Tensorboard
        self.global_step +=1

        if self.args.W[1] > 0:
            self.count_rw(self.label, out_patch_merge, 3)
        if self.args.W[0] > 0 and epo>self.args.fim_start_epo:
            self.count_rw(self.label, self.out_patch_hif_reduced, 1)
        if True:
            self.count_rw(self.label, self.out_patch, 0)   
    
    def aggregate_freq(self, epo):

        result_loss = {}
        patch_size = 16
        image_size = self.args.input_size

        '''If all class done, return'''
        if torch.sum(self.fim_ipc_stack) == 0:
            return True
    
        class_require = torch.zeros_like(self.fim_ipc_stack).to(self.dev)
        class_require[self.fim_ipc_stack > 0] = 1

        run_img_idx = torch.sum(self.label * class_require.view(1,self.num_class), dim=1) >= 1

        runned_num_per_class = torch.zeros((self.num_class)).to(self.dev)

        if torch.sum(run_img_idx) == 0:
            return False
        
        B = self.img.shape[0]
        B_real = torch.sum(run_img_idx)

        label_ones = torch.ones(B).to(self.dev)

        loss_matrix = torch.zeros(self.num_class,image_size//patch_size,image_size//patch_size).to(self.dev)
        criterion = nn.BCEWithLogitsLoss(reduction='sum')

        x1 = self.img[run_img_idx]      # Run only required class img
        sizex = x1.size()   # B 3 H W
        y1 = torch.zeros(sizex, dtype=torch.complex128).to(self.dev)
        
        mask = torch.ones((image_size,image_size)).to(self.dev)

        with torch.no_grad():
            for r in range(int(image_size/patch_size)):
            
                for c in range(int(image_size/patch_size/2)+1):
                    mask[mask==0] = 1

                    mask[patch_size*r:patch_size*(r+1),patch_size*c:patch_size*(c+1)] = 0
                    if int(image_size/patch_size)-r<int(image_size/patch_size) and int(image_size/patch_size)-c<int(image_size/patch_size):
                        mask[image_size-patch_size*(r):image_size-patch_size*(r-1),image_size-patch_size*(c):image_size-patch_size*(c-1)] = 0
                    
                    if r == 0 and c != 0:
                        mask[patch_size*r:patch_size*(r+1),image_size-patch_size*(c):image_size-patch_size*(c-1)] = 0
                    
                    x1 = self.img[run_img_idx]      # Run only required class img
                    sizex = x1.size()   # B 3 H W

                    # loss compute for existance / non-existance
                    y1 = torch.fft.fftshift(torch.fft.fft2(x1))
                    y1 = y1 * mask.view(1,1,mask.shape[0],mask.shape[1])

                    x1 = torch.fft.ifft2(torch.fft.ifftshift(y1))
                    x1 = torch.real(x1)
            
                    if B_real == 1:
                        # Dummy for running 2 gpu
                        x1 = x1.repeat(2,1,1,1)
                    
                    out_class, out_p_class = self.net_trm(x1, is_freq=True)     # B 20

                    if B_real == 1:
                        # Remove Dummy result
                        out_class = out_class[0].unsqueeze(0)
                        out_p_class = out_p_class[0].unsqueeze(0)

                    for class_idx in range(self.num_class):
                        class_img_num = int(torch.sum(self.label[run_img_idx, class_idx]))
                        need_img_num = int(self.fim_ipc_stack[class_idx])
                        if class_img_num <= need_img_num:
                            out_target = out_class[self.label[run_img_idx, class_idx] == 1][:, class_idx]     # img_num has class_idx
                            out_p_target = out_p_class[self.label[run_img_idx, class_idx] == 1][:, class_idx] # img_num has class_idx

                            runned_num_per_class[class_idx] = class_img_num
                        else:
                            out_target = out_class[self.label[run_img_idx, class_idx] == 1][:need_img_num][:, class_idx]     # img_num has class_idx
                            out_p_target = out_p_class[self.label[run_img_idx, class_idx] == 1][:need_img_num][:, class_idx] # img_num has class_idx

                            runned_num_per_class[class_idx] = need_img_num

                        target = label_ones[:int(runned_num_per_class[class_idx])]
                        
                        loss = criterion(out_target, target) + criterion(out_p_target, target) 
                        
                        loss_matrix[class_idx, r:(r+1), c:(c+1)] = loss
                        if int(image_size/patch_size)-r<int(image_size/patch_size) and int(image_size/patch_size)-c<int(image_size/patch_size):
                            loss_matrix[class_idx, image_size//patch_size-(r):image_size//patch_size-(r-1), image_size//patch_size-(c):image_size//patch_size-(c-1)] = loss

                        if r == 0 and c != 0:
                            loss_matrix[class_idx, r:(r+1), image_size//patch_size-(c):image_size//patch_size-(c-1)] = loss

            if epo not in self.loss_mat_for_freq:
                # Stack than interpolate
                self.loss_mat_for_freq[epo] = torch.zeros(self.num_class, image_size//patch_size, image_size//patch_size).to(self.dev)
            self.loss_mat_for_freq[epo] = self.loss_mat_for_freq[epo] + loss_matrix

            self.fim_ipc_stack -= runned_num_per_class
            
            if torch.sum(self.fim_ipc_stack) == 0:
                # All class Done
                return True
            else:
                return False
            

    def high_influence_freq_select_save(self, epo, hif_path, vis=False):
        image_size = self.args.input_size

        loss_mat = self.loss_mat_for_freq[epo]  # C H W

        '''Stack than interpolate'''
        loss_mat = F.interpolate(loss_mat.unsqueeze(0),[image_size,image_size],mode='bicubic',align_corners=False)
        loss_mat = loss_mat[0]
        
        np.save(osp.join(hif_path, f'LOSS_HIFM_{epo}.npy'), loss_mat.clone().cpu().numpy()) # C H W

        # Maximum loss during all epo
        if epo == self.args.fim_start_epo:
            self.max_loss = torch.zeros((self.num_class)).to(self.dev)
            
        loss_minmax_ratio = torch.zeros((self.num_class)).to(self.dev)
        loss_max_curall_ratio = torch.zeros((self.num_class)).to(self.dev)
        
        mask_of_rank_th = torch.zeros_like(loss_mat).to(self.dev)
        for c_idx in range(loss_mat.shape[0]):
            loss_mat_c = loss_mat[c_idx]

            loss_mat_c_min, loss_mat_c_max = loss_mat_c.min(), loss_mat_c.max()
            
            # Update maximum loss
            if self.max_loss[c_idx] < loss_mat_c_max:
                self.max_loss[c_idx] = loss_mat_c_max
                
            loss_max_curall_ratio[c_idx] = loss_mat_c_max / self.max_loss[c_idx]
            
            self.logger.info(f'class {c_idx} all loss max: {self.max_loss[c_idx]} / current loss max: {loss_mat_c_max}, min: {loss_mat_c_min}, ratio: {loss_mat_c_min / loss_mat_c_max}')
            loss_minmax_ratio[c_idx] = loss_mat_c_min / loss_mat_c_max
            
            mask_of_rank_th[c_idx] = (loss_mat_c - loss_mat_c_min) / (self.max_loss[c_idx] - loss_mat_c_min)

            '''If loss never calculated, use prev mask'''
            if loss_mat_c_max == 0:
                mask_of_rank_th[c_idx] = self.high_influence_freq_mask_prev[c_idx]
        
        mRatio = loss_minmax_ratio.mean()
        self.writer.add_scalar("mRatio/loss_min_over_max",mRatio,epo)

        mMaxRatio = loss_max_curall_ratio.mean()
        self.writer.add_scalar("mMaxRatio/loss_max_cur_over_all",mMaxRatio,epo)
        
        if epo == self.args.fim_start_epo:
            self.high_influence_freq_mask_prev = mask_of_rank_th
        else:
            self.high_influence_freq_mask_prev = self.high_influence_freq_mask_cur
        self.high_influence_freq_mask_cur = mask_of_rank_th
        
        mask_of_rank_th_np = mask_of_rank_th.cpu().numpy()     # C H W
        np.save(osp.join(hif_path, f'HIFM_{epo}_mask_lossmat.npy'), mask_of_rank_th_np) 

        if vis:
            hif_vis_path = osp.join(hif_path, f'vis')
            if not os.path.exists(hif_vis_path):
                os.makedirs(hif_vis_path)

            if self.num_class == 20:
                fig, axs = plt.subplots(4,5)
            if self.num_class == 80:
                fig, axs = plt.subplots(8,10)
            fig.set_figheight(15)
            fig.set_figwidth(15)

            if self.num_class == 20:
                for c_idx in range(mask_of_rank_th_np.shape[0]):
                    mask = mask_of_rank_th_np[c_idx]
                    axs[c_idx//5, c_idx%5].imshow(mask, cmap='gray')
                    axs[c_idx//5, c_idx%5].set_title(self.categories[c_idx])
                    axs[c_idx//5, c_idx%5].set_yticks([])
                    axs[c_idx//5, c_idx%5].set_xticks([])
            
            if self.num_class == 80:
                for c_idx in range(mask_of_rank_th_np.shape[0]):
                    mask = mask_of_rank_th_np[c_idx]
                    axs[c_idx//10, c_idx%10].imshow(mask, cmap='gray')
                    axs[c_idx//10, c_idx%10].set_title(self.categories[c_idx])
                    axs[c_idx//10, c_idx%10].set_yticks([])
                    axs[c_idx//10, c_idx%10].set_xticks([])
            
            plt.savefig(osp.join(hif_vis_path, f'HIFM_{epo}_vis.png'), bbox_inches='tight')
            plt.close()

            hif_used_path = osp.join(hif_path, f'used')
            if not os.path.exists(hif_used_path):
                os.makedirs(hif_used_path)

            if self.num_class == 20:
                fig, axs = plt.subplots(4,5)
            if self.num_class == 80:
                fig, axs = plt.subplots(8,10)
            fig.set_figheight(15)
            fig.set_figwidth(15)

            if self.num_class == 20:
                for c_idx in range(mask_of_rank_th_np.shape[0]):
                    mask = mask_of_rank_th[c_idx].clone()
                    mask = mask.cpu().numpy()
                    axs[c_idx//5, c_idx%5].imshow(mask, cmap='gray')
                    axs[c_idx//5, c_idx%5].set_title(self.categories[c_idx])
                    axs[c_idx//5, c_idx%5].set_yticks([])
                    axs[c_idx//5, c_idx%5].set_xticks([])
            
            if self.num_class == 80:
                for c_idx in range(mask_of_rank_th_np.shape[0]):
                    mask = mask_of_rank_th[c_idx].clone()
                    mask = mask.cpu().numpy()
                    axs[c_idx//10, c_idx%10].imshow(mask, cmap='gray')
                    axs[c_idx//10, c_idx%10].set_title(self.categories[c_idx])
                    axs[c_idx//10, c_idx%10].set_yticks([])
                    axs[c_idx//10, c_idx%10].set_xticks([])
            
            plt.savefig(osp.join(hif_used_path, f'HIFM_{epo}_used_vis.png'), bbox_inches='tight')
            plt.close()

    def vis_hifm_filter_mc(self, batch_step, epo, org_img, filtered_img, mask, org_cam_norm, filtered_cam_norm, label, name, img_name):
        '''
        org_img: 3 H W
        filtered_img 3 H W
        mask 20 H W
        org_cam 20 14 14 
        filtered_cam 20 14 14 
        label: 20
        '''
        vis_total_path = osp.join(self.args.exp_path, 'hif_filtered_cam')
        if not os.path.exists(vis_total_path):
            os.makedirs(vis_total_path)

        save_path = osp.join(vis_total_path, f'epo{epo}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if 'rcam' in img_name:
            save_path = osp.join(save_path, f'rcam')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        elif 'fcam' in img_name:
            save_path = osp.join(save_path, f'fcam')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        elif 'cam' in img_name:
            save_path = osp.join(save_path, f'cam')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
    
        org_img = denorm(org_img)
        filtered_img = denorm(filtered_img)

        exist_label = torch.nonzero(label)[:,0]
        label_num = exist_label.shape[0]
        
        fig = plt.figure(figsize=(40,20), dpi=100)
        axs = fig.add_subplot(3,label_num+1,1)
        axs.imshow(np.clip(org_img.cpu().detach().numpy().transpose(1,2,0) ,0,1))
        axs.set_title(f'{name}')
        axs.set_yticks([])
        axs.set_xticks([])

        axs = fig.add_subplot(3,label_num+1,label_num+1 + 1)
        axs.imshow(np.clip(filtered_img.cpu().detach().numpy().transpose(1,2,0) ,0,1))
        axs.set_title('filtered')
        axs.set_yticks([])
        axs.set_xticks([])

        for idx in range(label_num):
            target_class = int(exist_label[idx])
            org_img_cam = cam_on_image(org_img.cpu().detach().numpy(), org_cam_norm[target_class].detach().cpu().numpy())
            filtered_img_cam = cam_on_image(filtered_img.cpu().detach().numpy(), filtered_cam_norm[target_class].detach().cpu().numpy())

            axs = fig.add_subplot(3,label_num+1, 2 + idx)
            axs.imshow(org_img_cam.transpose(1,2,0))
            axs.set_title(f'org_img_cam_{self.categories[target_class]}')
            axs.set_yticks([])
            axs.set_xticks([])

            axs = fig.add_subplot(3,label_num+1,label_num+1 + 2 + idx)
            axs.imshow(filtered_img_cam.transpose(1,2,0))
            axs.set_title(f'filtered_img_cam_{self.categories[target_class]}')
            axs.set_yticks([])
            axs.set_xticks([])

            axs = fig.add_subplot(3,label_num+1,2*(label_num+1) + 2 + idx)
            axs.imshow(mask[target_class].cpu().detach().numpy(), cmap='gray')
            axs.set_title(f'hif_{self.categories[target_class]}')
            axs.set_yticks([])
            axs.set_xticks([])
        
        plt.savefig(osp.join(save_path, img_name), bbox_inches='tight')
        plt.close()

    # Initialization for msf-infer
    def infer_init(self,epo):
        n_gpus = torch.cuda.device_count()
        self.net_trm.eval()
        
    # (Multi-Thread) Infer MSF-CAM and save image/cam_dict/crf_dict
    def infer_multi(self, epo, val_path, dict_path, crf_path, vis=False, dict=False, crf=False, exp_path='.'):

        if self.phase != 'eval':
            self.set_phase('eval')

        epo_str = str(epo).zfill(3)
        gt = self.label[0].cpu().detach().numpy()
        self.gt_cls = np.nonzero(gt)[0]

        
        B, _, H, W = self.img.shape
        n_gpus = torch.cuda.device_count()

        with torch.no_grad():
            cam = self.net_trm.module.forward(self.img.cuda(),return_att=True,n_layers= 12)
            cam = F.interpolate(cam,[H,W],mode='bilinear',align_corners=False) * self.label.view(B,self.num_class,1,1)

            cam_flip = self.net_trm.module.forward(torch.flip(self.img,(3,)).cuda(),return_att=True,n_layers= 12)
            cam_flip = F.interpolate(cam_flip,[H,W],mode='bilinear',align_corners=False)*self.label.view(B,self.num_class,1,1)
            cam_flip = torch.flip(cam_flip,(3,))
    
            cam = cam+cam_flip
            norm_cam = self.max_norm(cam)[0].detach().cpu().numpy()

            self.cam_dict = {}

            for i in range(self.num_class):
                if self.label[0, i] > 1e-5:
                    self.cam_dict[i] = norm_cam[i]

            if vis:
                input = denorm(self.img[0])
                for c in self.gt_cls:
                    temp = cam_on_image(input.cpu().detach().numpy(), norm_cam[c])
                    self.writer.add_image(self.name+'/'+self.categories[c], temp, epo)

            if dict:
                np.save(osp.join(dict_path, self.name + '.npy'), self.cam_dict)

            if crf:
                for a in self.args.alphas:
                    crf_dict = _crf_with_alpha(self.cam_dict, self.name, alpha=a)
                    np.save(osp.join(crf_path, str(a).zfill(2), self.name + '.npy'), crf_dict)
    
    def denormforDiff(self,img):
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        mean_tensor = torch.tensor(imagenet_mean).reshape(1, 3, 1, 1).cuda()
        std_tensor = torch.tensor(imagenet_std).reshape(1, 3, 1, 1).cuda()

        denorm_img = img * std_tensor + mean_tensor

        return denorm_img
    
    def normforCls(self,img):
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        mean_tensor = torch.tensor(imagenet_mean).reshape(1, 3, 1, 1).cuda()
        std_tensor = torch.tensor(imagenet_std).reshape(1, 3, 1, 1).cuda()

        denorm_img = (img - mean_tensor) / std_tensor

        return denorm_img


    # Print loss/accuracy (and re-initialize them)
    def print_log(self, epo, iter):

        loss_str = ''
        acc_str = ''

        for i in range(len(self.loss_names)):
            loss_str += self.loss_names[i] + ' : ' + str(round(self.running_loss[i] / self.count, 5)) + ', '

        for i in range(len(self.acc_names)):
            if self.right_count[i] != 0:
                acc = 100 * self.right_count[i] / (self.right_count[i] + self.wrong_count[i])
                acc_str += self.acc_names[i] + ' : ' + str(round(acc, 2)) + ', '
                self.accs[i] = acc

        self.logger.info(loss_str[:-2])
        self.logger.info(acc_str[:-2])

        ###Tensorboard###
        for i in range(len(self.loss_names)):
            self.writer.add_scalar("Loss/%s"%self.loss_names[i],(self.running_loss[i] / self.count),self.global_step)

        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.count = 0

    def count_rw(self, label, out, idx):
        for b in range(out.size(0)):  # 8
            gt = label[b].cpu().detach().numpy()
            gt_cls = np.nonzero(gt)[0]
            num = len(np.nonzero(gt)[0])
            pred = out[b].cpu().detach().numpy()
            pred_cls = pred.argsort()[-num:][::-1]

            for c in gt_cls:
                if c in pred_cls:
                    self.right_count[idx] += 1
                else:
                    self.wrong_count[idx] += 1

    @torch.no_grad()
    def mvweight(self):
        for param_main, param_sup in zip(self.net_main.parameters(), self.net_sup.parameters()):
            # param_sup.data = self.M * param_sup.data + (1 - self.M) * param_main.data
            param_sup.data =  param_main.data

    # Max_norm
    def max_norm(self, cam_cp):
        N, C, H, W = cam_cp.size()
        cam_cp = F.relu(cam_cp)
        max_v = torch.max(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        min_v = torch.min(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        cam_cp = F.relu(cam_cp - min_v - 1e-5) / (max_v - min_v + 1e-5)
        return cam_cp
    
    def cam_l1(self, cam1, cam2):
        return torch.mean(torch.abs(cam2.detach() - cam1))

    def split_label(self):

        bs = self.label.shape[0] if self.phase == 'train' else 1
        self.label_exist = torch.zeros(bs, 20).cuda()
        for i in range(bs):
            label_idx = torch.nonzero(self.label[i], as_tuple=False)
            rand_idx = torch.randint(0, len(label_idx), (1,))
            target = label_idx[rand_idx][0]
            self.label_exist[i, target] = 1
        self.label_remain = self.label - self.label_exist

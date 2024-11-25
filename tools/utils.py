import os
import os.path as osp
import random

import PIL

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import voc12.data
import tools.imutils as imutils
from timm.data import create_transform

class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

class PolyOptimizer_adam(torch.optim.Adam):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


class PolyOptimizer_cls(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                if i == 4:
                    self.param_groups[i]['lr'] = self.__initial_lr[i]
                else:
                    self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

def make_path(args):

    exp_path = osp.join('./experiments', args.name)
    ckpt_path = osp.join(exp_path, 'ckpt')
    train_path = osp.join(exp_path, 'train')
    val_path = osp.join(exp_path, 'val')
    infer_path = osp.join(exp_path, 'infer')
    dict_path = osp.join(exp_path, 'dict')
    crf_path = osp.join(exp_path, 'crf')
        
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        os.makedirs(ckpt_path)
        os.makedirs(train_path)
        os.makedirs(val_path)
        os.makedirs(infer_path)
        os.makedirs(dict_path)
        os.makedirs(crf_path)
        print(exp_path + ' is built.')
    else:
        print(exp_path + ' already exsits.')

    for alpha in args.alphas:
        crf_alpha_path = osp.join(crf_path, str(alpha).zfill(2))
        if not os.path.exists(crf_alpha_path):
            os.makedirs(crf_alpha_path)

    return exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path


def make_path_with_log(args):

    exp_path = osp.join('./experiments', args.name)
    ckpt_path = osp.join(exp_path, 'ckpt')
    train_path = osp.join(exp_path, 'train')
    val_path = osp.join(exp_path, 'val')
    infer_path = osp.join(exp_path, 'infer')
    dict_path = osp.join(exp_path, 'dict')
    crf_path = osp.join(exp_path, 'crf')
    log_path = osp.join(exp_path, 'log.txt')
        
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        os.makedirs(ckpt_path)
        os.makedirs(train_path)
        os.makedirs(val_path)
        os.makedirs(infer_path)
        os.makedirs(dict_path)
        os.makedirs(crf_path)
        print(exp_path + ' is built.')
    else:
        print(exp_path + ' already exsits.')

    for alpha in args.alphas:
        crf_alpha_path = osp.join(crf_path, str(alpha).zfill(2))
        if not os.path.exists(crf_alpha_path):
            os.makedirs(crf_alpha_path)

    return exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path, log_path


def build_dataset_trm(is_train, args, gen_attn=False):
    dataset = None
    nb_classes = None

    if args.data_set == 'VOC12':
        args.gen_attention_maps = gen_attn
        transform = build_transform(is_train, args)

        if not gen_attn:
            # Train / FIM
            dataset = voc12.data.VOC12Dataset(img_name_list_path=args.train_list, voc12_root=args.data_path,
                            train=is_train, gen_attn=gen_attn, transform=transform)
        else:
            # Val
            dataset = voc12.data.VOC12Dataset(img_name_list_path=args.val_list, voc12_root=args.data_path,
                            train=is_train, gen_attn=gen_attn, transform=transform)
            
        nb_classes = 20

    elif args.data_set == 'COCO':
        args.gen_attention_maps = gen_attn
        transform = build_transform(is_train, args)
    
        if not gen_attn:
            # Train / FIM
            dataset = voc12.data.COCOClsDataset(img_name_list_path=args.train_list, coco_root=args.data_path,label_file_path=args.label_file_path,
                            train=is_train, gen_attn=gen_attn, transform=transform)
        else:
            # Validation
            dataset = voc12.data.COCOClsDataset(img_name_list_path=args.val_list, coco_root=args.data_path,label_file_path=args.label_file_path,
                            train=is_train, gen_attn=gen_attn, transform=transform)
    
        nb_classes = 80

    return dataset, nb_classes

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms.functional import InterpolationMode

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            scale=args.scale
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)

        return transform

    t = []
    if resize_im and not args.gen_attention_maps:
        t.append(
            transforms.Resize((args.input_size, args.input_size))
        )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
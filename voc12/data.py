import pdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import PIL.Image
import os.path
import scipy.misc
import random

import tools.imutils as imutils

##########################################################
from scipy.spatial.distance import cdist
import scipy.ndimage
import numpy as np
from matplotlib import pyplot as plt
##########################################################


IMG_FOLDER_NAME = "JPEGImages" ##################caution
ANNOT_FOLDER_NAME = "Annotations"

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

def save_img(x, path):
    plt.imshow(x)
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(path)
    plt.close()

def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME,img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab

def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def load_image_label_list_from_npy(img_name_list,coco = False):

    if coco :
        cls_labels_dict = np.load('coco14/cls_labels_coco.npy',allow_pickle=True).item()
        cls_labels_dict_new = {}
        for k,v in cls_labels_dict.items():
            new_key = "%012d"%k
            cls_labels_dict_new[new_key]=v

        return [cls_labels_dict_new[img_name] for img_name in img_name_list]
    else:
        cls_labels_dict = np.load('voc12/cls_labels.npy',allow_pickle=True).item()
        return [cls_labels_dict[img_name] for img_name in img_name_list]
    
    # cls_labels_dict = np.load('voc12/cls_labels_custom2.npy', allow_pickle=True).item()

def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

def load_img_name_list(dataset_path, coco=False):
    if coco:
        img_name_list = open(dataset_path).read().splitlines()
        # pdb.set_trace()
        # img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]

    else:
        img_gt_name_list = open(dataset_path).read().splitlines()
        img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]

    return img_name_list

class COCOClsDataset(Dataset):
    def __init__(self, img_name_list_path, coco_root, label_file_path, train=True, transform=None, transform2=None, gen_attn=False):
        self.img_name_list = load_img_name_list(img_name_list_path, coco=True)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, coco=True)
        self.coco_root = coco_root
        self.transform = transform
        self.transform2 = transform2
        self.train = train
        self.gen_attn = gen_attn

        if 'train' in img_name_list_path:
            self.dataset_train = True

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        # if self.train or self.gen_attn:
        if self.dataset_train:
            img = PIL.Image.open(os.path.join(self.coco_root, 'train2014',"COCO_train2014_"+name + '.jpg')).convert("RGB")
        else:
            img = PIL.Image.open(os.path.join(self.coco_root, 'val2014',"COCO_val2014_"+name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        if self.transform:
            img = self.transform(img)
            
        if self.transform2:
            img_diff = self.transform2(img_diff)
        else:
            img_diff = self.transform(img_diff)
        
        return img_diff, label, name

    def __len__(self):
        return len(self.img_name_list)


from torchvision import transforms

    
class VOC12Dataset(Dataset):
    def __init__(self, img_name_list_path, voc12_root, train=True, transform=None, transform2=None, gen_attn=False):
        # img_name_list_path = os.path.join(img_name_list_path, f'{"train_aug" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.voc12_root = voc12_root
        self.transform = transform
        self.transform2 = transform2
        
    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        img_diff = img

        if self.transform:
            img = self.transform(img)
            
        if self.transform2:
            img_diff = self.transform2(img_diff)
        else:
            img_diff = self.transform(img_diff)
        
        return img_diff, label, name

    def __len__(self):
        return len(self.img_name_list)

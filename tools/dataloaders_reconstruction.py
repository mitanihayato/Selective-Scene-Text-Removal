from itertools import count
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import re
import glob
from os.path import join


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def pair(dir_input):
    '''
        dir_input:input path

        output: [[text0.png, background0.png, scene0.png], 
                       [text3.png, background3.png, scene3.png], 
                       [text10.png, background10.png, scene10.png], ...]
    '''

    img_list = []
    files = sorted(glob.glob(join(dir_input, "*.png")), key=natural_keys)

    for i in range(len(files)):
        input_img_name = files[i].replace((dir_input + '/'), '')
        img_num = re.sub(r"\D", "", input_img_name)
        input2_img_name = 'background' + img_num + '.png'
        label_img_name = 'scene' + img_num + '.png'
        pair = [input_img_name, input2_img_name, label_img_name]
        img_list.append(pair)
    
    return img_list


def resize_w_pad(img, target_w, target_h):
    '''
    Resize PIL image while maintaining aspect ratio
    '''
    # get new size
    target_ratio = target_h / target_w
    img_ratio = img.size(1) / img.size(2) 
    if target_ratio > img_ratio:
        # fixed with width
        new_w = target_w
        new_h = round(new_w * img_ratio)
    else:
        # fixed with height
        new_h = target_h
        new_w = round(new_h / img_ratio)
    # resize to new size
    tr = transforms.Resize((new_h, new_w))
    img = tr(img)
    
    # padding to target size
    horizontal_pad = (target_w - new_w) / 2
    vertical_pad = (target_h - new_h) / 2
    left = horizontal_pad if horizontal_pad % 1 == 0 else horizontal_pad + 0.5
    right = horizontal_pad if horizontal_pad % 1 == 0 else horizontal_pad - 0.5
    top = vertical_pad if vertical_pad % 1 == 0 else vertical_pad + 0.5
    bottom = vertical_pad if vertical_pad % 1 == 0 else vertical_pad - 0.5

    padding = (int(left), int(top), int(right), int(bottom))
    img = transforms.Pad(padding)(img)

    return img


class SynthTextDataset(Dataset):

    def __init__(self, im_list, input_root_dir, input2_root_dir, label_root_dir, target_size, transform=None):
        self.data_name_list = im_list
        self.input_root_dir = input_root_dir
        self.input2_root_dir = input2_root_dir
        self.label_root_dir = label_root_dir
        self.transform = transform
        self.target_size = target_size
    
    def __len__(self):
        return len(self.data_name_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img_name = os.path.join(self.input_root_dir,
                                self.data_name_list[idx][0])
        input2_img_name = os.path.join(self.input2_root_dir,
                                self.data_name_list[idx][1])
        label_img_name = os.path.join(self.label_root_dir,
                                self.data_name_list[idx][2])

        input_image = torch.from_numpy(io.imread(input_img_name).transpose(2,0,1))
        input2_image = torch.from_numpy(io.imread(input2_img_name).transpose(2,0,1))
        label_image = torch.from_numpy(io.imread(label_img_name).transpose(2,0,1))

        if self.transform:
             input_image = resize_w_pad(input_image, self.target_size, self.target_size) / 255.
             input2_image = resize_w_pad(input2_image, self.target_size, self.target_size) / 255.
             label_image = resize_w_pad(label_image, self.target_size, self.target_size) / 255.

        return input_image, input2_image, label_image
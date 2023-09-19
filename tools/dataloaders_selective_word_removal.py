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
import csv


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def pair(dir_label, file):
    '''
        dir_input: input path

        output : [[text1.png, text_removea1.png, 0], 
                      [text1.png, text_removeb1.png, 1], 
                      [text1.png, text_removec1.png, 3], 
                      [text3.png, text_removea3.png, 2], 
                      [text3.png, text_removeb3.png, 6], 
                      [text3.png, text_removec3.png, 9], ...]
    '''

    img_list = []
    files = sorted(glob.glob(join(dir_label, "*.png")), key=natural_keys)

    with open(file) as f:
        reader = csv.reader(f)
        l = [row for row in reader]
        condition_list = [list(x) for x in zip(*l)]

        for i in range(len(files)):
            label_img_name = files[i].replace((dir_label + '/'), '')
            if 'remove_a' in label_img_name:
                img_num = re.sub(r"\D", "", label_img_name)
                check_img_name = 'text_remove_a' + img_num
                for j in range(len(condition_list[0])):
                    if check_img_name == condition_list[0][j]:
                        condition_num = int(condition_list[1][j])
                        break
                input_img_name = 'text' + img_num + '.png'
                pair = [input_img_name, label_img_name, condition_num]
                img_list.append(pair)

            elif 'remove_b' in label_img_name:
                img_num = re.sub(r"\D", "", label_img_name)
                check_img_name = 'text_remove_b' + img_num
                for j in range(len(condition_list[0])):
                    if check_img_name == condition_list[0][j]:
                        condition_num = int(condition_list[1][j])
                        break
                input_img_name = 'text' + img_num + '.png'
                pair = [input_img_name, label_img_name, condition_num]
                img_list.append(pair)

            else:
                img_num = re.sub(r"\D", "", label_img_name)
                check_img_name = 'text_remove_c' + img_num
                for j in range(len(condition_list[0])):
                    if check_img_name == condition_list[0][j]:
                        condition_num = int(condition_list[1][j])
                        break
                input_img_name = 'text' + img_num + '.png'
                pair = [input_img_name, label_img_name, condition_num]
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

    def __init__(self, im_list, input_root_dir, label_root_dir, target_size, data_name, transform=None):
        self.data_name_list = im_list
        self.input_root_dir = input_root_dir
        self.label_root_dir = label_root_dir
        self.transform = transform
        self.target_size = target_size
        self.target_names = data_name
    
    def __len__(self):
        return len(self.data_name_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img_name = os.path.join(self.input_root_dir,
                                self.data_name_list[idx][0])
        label_img_name = os.path.join(self.label_root_dir,
                                self.data_name_list[idx][1])

        input_image = torch.from_numpy(io.imread(input_img_name).transpose(2,0,1))
        label_image = torch.from_numpy(io.imread(label_img_name).transpose(2,0,1))

        target_offset = self.data_name_list[idx][2]
        input_condition = np.zeros(len(self.target_names), dtype=np.float32)
        input_condition[target_offset] = 1.
        input_condition = torch.from_numpy(input_condition)

        if self.transform:
             input_image = resize_w_pad(input_image, self.target_size, self.target_size) / 255.
             label_image = resize_w_pad(label_image, self.target_size, self.target_size) / 255.

        return input_image, label_image, input_condition
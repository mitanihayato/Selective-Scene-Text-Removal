'''
    Text Extraction Module test code
'''


from itertools import count
import os
import torch
import torch.utils.data
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from tools.pytorchtools import EarlyStopping
import re
from tools import dataloaders_text_extraction
from models.Text_extraction_module.text_extra_module import text_extraction_module
from collections import OrderedDict
import argparse


# GPU check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
print(device)

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)


def get_args():
    parser = argparse.ArgumentParser(description='sample',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--model_path', type=str, default='./train_model/text_extraction_module/checkpoint_model.pth', help='model path')
    parser.add_argument('--input1', type=str, default='./sample_data/background_test', help='input(background) path')
    parser.add_argument('--input2', type=str, default='./sample_data/scene_test', help='input(dcene) path')
    parser.add_argument('--label', type=str, default='./sample_data/text_test', help='label path')
    parser.add_argument('--img_size', type=int, default=512, help='Image size')

    return parser.parse_args()




def test(net, dataloader):

    new_save_dir_input = './test_result/text_extraction_module/input/'
    new_save_dir_input2 = './test_result/text_extraction_module/input2/'
    new_save_dir_output = './test_result/text_extraction_module/output/'
    os.makedirs(new_save_dir_input, exist_ok=True)
    os.makedirs(new_save_dir_input2, exist_ok=True)
    os.makedirs(new_save_dir_output, exist_ok=True)

    with torch.no_grad():

        i = 0
        net.eval()

        for data in tqdm(dataloader):
            inputs, inputs2, labels = data
            inputs = inputs.to(device)
            inputs2 = inputs2.to(device)

            outputs = net(inputs, inputs2)

            img_name_input = new_save_dir_input +'/input_back_image' + str(i) + '.png'
            img_name_input2 = new_save_dir_input2 +'/input_scene_image' + str(i) + '.png'
            img_name_output = new_save_dir_output + '/output_image' + str(i) + '.png'
            utils.save_image(inputs, img_name_input, normalize=True)
            utils.save_image(inputs2, img_name_input2, normalize=True)
            utils.save_image(outputs, img_name_output, normalize=True)
            i+=1 

    
    print('Finished Testing')


if __name__ == '__main__':
    args = get_args()

    input_root_dir = args.input1
    input2_root_dir = args.input2
    label_root_dir = args.label

    im_list = dataloaders_text_extraction.pair(input_root_dir)
    target_size = args.img_size
    dataset = dataloaders_text_extraction.SynthTextDataset(im_list, input_root_dir, input2_root_dir, label_root_dir, target_size, dataloaders_text_extraction.resize_w_pad)
    dataloader = DataLoader(dataset=dataset, batch_size = args.batch, shuffle=False, num_workers=4)

    path = args.model_path
    state_dict = torch.load(path)
    net = text_extraction_module(n_channels=6, n_classes=4)
    net.load_state_dict(state_dict)

    net.to(device)

    test(net, dataloader)
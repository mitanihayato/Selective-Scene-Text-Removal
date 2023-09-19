'''
    Background Extraction Module test code
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
import tools.dataloaders_background_extra as dataloaders_background_extra
from models.Background_extraction_module.backgaround_extra_module import background_extraction_module
from collections import OrderedDict
import argparse


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
    parser.add_argument('--model_path', type=str, default='./train_model/background_extraction_module/checkpoint_model.pth', help='model path')
    parser.add_argument('--input', type=str, default='./sample_data/scene_test', help='input path')
    parser.add_argument('--label', type=str, default='./sample_data/background_test', help='label path')
    parser.add_argument('--img_size', type=int, default=512, help='Image size')

    return parser.parse_args()



def test(net, dataloader):

    new_save_dir_input = './test_result/background_extraction_module/input/'
    new_save_dir_output = './test_result/background_extraction_module/output/'
    os.makedirs(new_save_dir_input, exist_ok=True)
    os.makedirs(new_save_dir_output, exist_ok=True)

    with torch.no_grad():

        i = 0
        net.eval()

        for data in tqdm(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)

            outputs = net(inputs)

            img_name_input = new_save_dir_input +'/input_image' + str(i) + '.png'
            img_name_output = new_save_dir_output + '/output_image' + str(i) + '.png'
            utils.save_image(inputs, img_name_input, normalize=True)
            utils.save_image(outputs, img_name_output, normalize=True)
            i+=1 

    
    print('Finished Testing')


if __name__ == '__main__':
    args = get_args()
    input_root_dir = args.input
    label_root_dir = args.label

    im_list = dataloaders_background_extra.pair(input_root_dir)
    target_size = 512
    dataset = dataloaders_background_extra.SynthTextDataset(im_list, input_root_dir, label_root_dir, target_size, dataloaders_background_extra.resize_w_pad)
    dataloader = DataLoader(dataset=dataset, batch_size = args.batch, shuffle=False, num_workers=4)

    path = args.model_path
    state_dict = torch.load(path)
    net = background_extraction_module(in_channels=3)
    net.load_state_dict(state_dict)

    net.to(device)

    test(net, dataloader)
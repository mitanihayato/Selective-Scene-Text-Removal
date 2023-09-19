'''
    Selective Word Removal Module test code
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
import tools.dataloaders_selective_word_removal as dataloaders_selective_word_removal
from models.Selective_word_removal_module.cunet_model import CUNET
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
    parser.add_argument('--model_path', type=str, default='./train_model/selective_word_removal_module/checkpoint_model.pth', help='model path')
    parser.add_argument('--input', type=str, default='./sample_data/text_test', help='input path')
    parser.add_argument('--label', type=str, default='./sample_data/text_remove_test', help='label path')
    parser.add_argument('--condition', type=str, default='./sample_data/condition_text.csv', help='condition path')
    parser.add_argument('--remove_words', type=list, default=['China', 'France', 'Germany', 'India', 'Japan'], help='list of removal words')
    parser.add_argument('--img_size', type=int, default=512, help='Image size')

    return parser.parse_args()





def test(net, dataloader, data_num, data_kind):
    save_dir_input = []
    save_dir_output = []

    save_img_num = []
    for i in range(data_num):
        save_img_num.append(0)

    for l in range(len(data_kind)):
        new_save_dir_input_train = './test_result/selective_word_removal_module/input/' + data_kind[l]
        new_save_dir_output_train =  './test_result/selective_word_removal_module/output/' + data_kind[l]
        save_dir_input.append(new_save_dir_input_train)
        save_dir_output.append(new_save_dir_output_train)
        os.makedirs(new_save_dir_input_train, exist_ok=True)
        os.makedirs(new_save_dir_output_train, exist_ok=True)


    with torch.no_grad():
        net.eval()
        for data in tqdm(dataloader):
            inputs, labels, input_condition = data
            inputs = inputs.to(device)
            input_condition = input_condition.to(device)
            outputs = net(inputs, input_condition)

            for k in range(int(torch.numel(input_condition)/data_num)):
                for j in range(data_num):
                    if input_condition[k][j].item() == 1.0:
                        condition_num = j
                        break

                img_name_input = save_dir_input[condition_num] +'/input_image' + str(save_img_num[condition_num]) + '.png'
                img_name_outputs = save_dir_output[condition_num] + '/output_image' + str(save_img_num[condition_num]) + '.png'
                utils.save_image(inputs[k], img_name_input, normalize=True)
                utils.save_image(outputs[k], img_name_outputs, normalize=True)

                save_img_num[condition_num]+=1

    print('Finished Testing')




if __name__ == '__main__':
    args = get_args()

    input_root_dir = args.input
    label_root_dir = args.label
    condition_file = args.condition
    data_kind = args.remove_words
    data_num = len(data_kind)

    im_list = dataloaders_selective_word_removal.pair(label_root_dir, condition_file)
    target_size = args.img_size
    dataset = dataloaders_selective_word_removal.SynthTextDataset(im_list, input_root_dir, label_root_dir, target_size, data_kind, dataloaders_selective_word_removal.resize_w_pad)
    dataloader = DataLoader(dataset=dataset, batch_size = args.batch, shuffle=False, num_workers=4)

    path = args.model_path
    state_dict = torch.load(path)
    net = CUNET(n_layers=6, input_channels=4, filters_layer_1=64, film_type='complex', control_input_dim=data_num)
    net.load_state_dict(state_dict)
    net.to(device)

    test(net, dataloader, data_num, data_kind)
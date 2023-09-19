'''
    Selective Scene Text Removal fine-tuning test code
'''

from itertools import count
import os
from turtle import width
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
from tools.pytorchtools_finetuning import EarlyStopping
import re
import tools.dataloaders_finetuning_test as dataloaders_finetuning_test
from models.Background_extraction_module.backgaround_extra_module import background_extraction_module
from models.Text_extraction_module.text_extra_module import text_extraction_module
from models.Selective_word_removal_module.cunet_model import CUNET
from models.Reconstruction_module.reconst_module import reconstruction_module
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
    parser.add_argument('--epoch', type=int, default=10000, help='Number of epoch')
    parser.add_argument('--input', type=str, default='./sample_data/scene_test', help='input path')
    parser.add_argument('--label', type=str, default='./sample_data/remove_test', help='label path')
    parser.add_argument('--condition', type=str, default='./sample_data/condition_scene.csv', help='condition path')
    parser.add_argument('--remove_words', type=list, default=['China', 'France', 'Germany', 'India', 'Japan'], help='list of removal words')
    parser.add_argument('--model_path', type=str, default='./train_model/finetuning/checkpoint_model.pth', help='model path')
    parser.add_argument('--img_size', type=int, default=512, help='Image size')
    parser.add_argument('--early_stopping', type=int, default=50, help='Early stopping epoch')

    return parser.parse_args()



def test(background_extraction, text_extraction, selective_word_removal, reconstruction, dataloader, data_num, data_kind):
    background_extraction.eval()
    text_extraction.eval()
    selective_word_removal.eval()
    reconstruction.eval()  

    save_dir_input = []
    save_dir_output_background_extraction = []
    save_dir_output_text_extraction = []
    save_dir_output_selective_word_removal = []
    save_dir_output_reconstruction = []

    save_img_num = []
    for i in range(data_num):
        save_img_num.append(0)

    for l in range(len(data_kind)):
        new_save_dir_input = './test_result/finetuning/input/' + data_kind[l]
        new_save_dir_outputs_background_extraction = './test_result/finetuning/output_background_extraction/' + data_kind[l]
        new_save_dir_outputs_text_extraction = './test_result/finetuning/output_text_extraction/' + data_kind[l]
        new_save_dir_outputs_selective_word_removal = './test_result/finetuning/output_selective_word_removal/' + data_kind[l]
        new_save_dir_outputs_reconstruction = './test_result/finetuning/output_reconstruction/' + data_kind[l]
        
        save_dir_input.append(new_save_dir_input)
        save_dir_output_background_extraction.append(new_save_dir_outputs_background_extraction)
        save_dir_output_text_extraction.append(new_save_dir_outputs_text_extraction)
        save_dir_output_selective_word_removal.append(new_save_dir_outputs_selective_word_removal)
        save_dir_output_reconstruction.append(new_save_dir_outputs_reconstruction)

        os.makedirs(new_save_dir_input, exist_ok=True)
        os.makedirs(new_save_dir_outputs_background_extraction, exist_ok=True)
        os.makedirs(new_save_dir_outputs_text_extraction, exist_ok=True)
        os.makedirs(new_save_dir_outputs_selective_word_removal, exist_ok=True)
        os.makedirs(new_save_dir_outputs_reconstruction, exist_ok=True)

    with torch.no_grad():

        for data in tqdm(dataloader, leave=False):
            inputs, labels, conditions = data
            inputs = inputs.to(device)
            conditions = conditions.to(device)

            outputs_background_extraction = background_extraction(inputs)
            outputs_text_extraction = text_extraction(outputs_background_extraction, inputs)
            outputs_selective_word_removal = selective_word_removal(outputs_text_extraction, conditions)
            outputs_reconstruction = reconstruction(outputs_selective_word_removal, outputs_background_extraction)

            for k in range(int(torch.numel(conditions)/data_num)):
                for j in range(data_num):
                    if conditions[k][j].item() == 1.0:
                        condition_num = j
                        break

                img_name_input = save_dir_input[condition_num] +'/input' + str(save_img_num[condition_num]) + '.png'
                img_name_outputs_background_extraction = save_dir_output_background_extraction[condition_num] + '/output' + str(save_img_num[condition_num]) + '.png'
                img_name_outputs_text_extraction = save_dir_output_text_extraction[condition_num] + '/output' + str(save_img_num[condition_num]) + '.png'
                img_name_outputs_selective_word_removal = save_dir_output_selective_word_removal[condition_num] + '/output' + str(save_img_num[condition_num]) + '.png'
                img_name_outputs_reconstruction = save_dir_output_reconstruction[condition_num] + '/output' + str(save_img_num[condition_num]) + '.png'
                utils.save_image(inputs[k], img_name_input, normalize=True)
                utils.save_image(outputs_background_extraction[k], img_name_outputs_background_extraction, normalize=True)
                utils.save_image(outputs_text_extraction[k], img_name_outputs_text_extraction, normalize=True)
                utils.save_image(outputs_selective_word_removal[k], img_name_outputs_selective_word_removal, normalize=True)
                utils.save_image(outputs_reconstruction[k], img_name_outputs_reconstruction, normalize=True)

                save_img_num[condition_num]+=1

    print('Finished Testing')



if __name__ == '__main__':
    args = get_args()

    input_root_dir = args.input
    label_root_dir = args.label
    condition_file = args.condition
    data_kind = args.remove_words
    data_num = len(data_kind)

    im_list = dataloaders_finetuning_test.pair(label_root_dir, condition_file)
    target_size = args.img_size
    dataset = dataloaders_finetuning_test.SynthTextDataset(im_list, input_root_dir, label_root_dir, target_size, data_kind, dataloaders_finetuning_test.resize_w_pad)
    dataloader = DataLoader(dataset=dataset, batch_size = args.batch, shuffle=False, num_workers=4)

    checkpoint = torch.load(args.model_path)
    background_extraction = background_extraction_module(in_channels=3)  
    background_extraction.load_state_dict(checkpoint['background_extraction_module'])

    text_extraction = text_extraction_module(n_channels=6, n_classes=4)
    text_extraction.load_state_dict(checkpoint['text_extraction_module'])

    selective_word_removal = CUNET(n_layers=6, input_channels=4, filters_layer_1=64, film_type='complex', control_input_dim=data_num)
    selective_word_removal.load_state_dict(checkpoint['selective_word_removal_module'])

    reconstruction = reconstruction_module(n_channels=7, n_classes=3)
    reconstruction.load_state_dict(checkpoint['reconstruction_module'])

    background_extraction.to(device)
    text_extraction.to(device)
    selective_word_removal.to(device)
    reconstruction.to(device)

    test(background_extraction, text_extraction, selective_word_removal, reconstruction, dataloader, data_num, data_kind)
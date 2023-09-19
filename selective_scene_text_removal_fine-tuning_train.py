'''
    Selective Scene Text Removal fine-tuning training code
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
import tools.dataloaders_finetuning as dataloaders_finetuning
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
    
    parser.add_argument('--batch', type=int, default=3, help='Batch size')
    parser.add_argument('--epoch', type=int, default=10000, help='Number of epoch')
    parser.add_argument('--input', type=str, default='./sample_data/scene_train', help='input path')
    parser.add_argument('--label1', type=str, default='./sample_data/background_train', help='label(background) path')
    parser.add_argument('--label2', type=str, default='./sample_data/remove_train', help='label(remove word image) path')
    parser.add_argument('--condition', type=str, default='./sample_data/condition_scene.csv', help='condition path')
    parser.add_argument('--remove_words', type=list, default=['China', 'France', 'Germany', 'India', 'Japan'], help='list of removal words')
    parser.add_argument('--background_extraction_path', type=str, default='./train_model/background_extraction_module/checkpoint_model.pth', help='background extraction module model path')
    parser.add_argument('--text_extraction_path', type=str, default='./train_model/text_extraction_module/checkpoint_model.pth', help='text extraction module model path')
    parser.add_argument('--selective_word_removal_path', type=str, default='./train_model/selective_word_removal_module/checkpoint_model.pth', help='selective word removal module model path')
    parser.add_argument('--reconstruction_path', type=str, default='./train_model/reconstruction_module/checkpoint_model.pth', help='reconstruction module model path')
    parser.add_argument('--img_size', type=int, default=512, help='Image size')
    parser.add_argument('--early_stopping', type=int, default=50, help='Early stopping epoch')

    return parser.parse_args()


# validation
def validate(background_extraction, text_extraction, selective_word_removal, reconstruction, dataloader, dataset, criterion, epoch, data_num, data_kind, save_dir_inputs, save_dir_outputs_background_extraction, save_dir_outputs_text_extraction, save_dir_outputs_selective_word_removal, save_dir_outputs_reconstruction, save_dir_labels):
    background_extraction.eval()
    text_extraction.eval()
    selective_word_removal.eval()
    reconstruction.eval()  
    with torch.no_grad():
        total_loss = 0.0
        total_loss_background_extraction= 0.0
        total_loss_reconstruction = 0.0 

        if ((epoch+1)%100) == 0:

            save_dir_input = []
            save_dir_output_background_extraction = []
            save_dir_output_text_extraction = []
            save_dir_output_selective_word_removal = []
            save_dir_output_reconstruction = []
            save_dir_label = []

            save_img_num = []
            for i in range(data_num):
                save_img_num.append(0)

            for l in range(len(data_kind)):
                new_save_dir_input_val = save_dir_inputs + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                new_save_dir_outputs_background_extraction_val = save_dir_outputs_background_extraction + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                new_save_dir_outputs_text_extraction_val = save_dir_outputs_text_extraction + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                new_save_dir_outputs_selective_word_removal_val = save_dir_outputs_selective_word_removal + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                new_save_dir_outputs_reconstruction_val = save_dir_outputs_reconstruction + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                new_save_dir_label_val = save_dir_labels + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                save_dir_input.append(new_save_dir_input_val)
                save_dir_output_background_extraction.append(new_save_dir_outputs_background_extraction_val)
                save_dir_output_text_extraction.append(new_save_dir_outputs_text_extraction_val)
                save_dir_output_selective_word_removal.append(new_save_dir_outputs_selective_word_removal_val)
                save_dir_output_reconstruction.append(new_save_dir_outputs_reconstruction_val)
                save_dir_label.append(new_save_dir_label_val)
                os.makedirs(new_save_dir_input_val, exist_ok=True)
                os.makedirs(new_save_dir_outputs_background_extraction_val, exist_ok=True)
                os.makedirs(new_save_dir_outputs_text_extraction_val, exist_ok=True)
                os.makedirs(new_save_dir_outputs_selective_word_removal_val, exist_ok=True)
                os.makedirs(new_save_dir_outputs_reconstruction_val, exist_ok=True)
                os.makedirs(new_save_dir_label_val, exist_ok=True)


        for data in dataloader:
            inputs, labels, labels2, conditions = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels2 = labels2.to(device)
            conditions = conditions.to(device)

            outputs_background_extraction = background_extraction(inputs)
            outputs_text_extraction = text_extraction(outputs_background_extraction, inputs)
            outputs_selective_word_removal = selective_word_removal(outputs_text_extraction, conditions)
            outputs_reconstruction = reconstruction(outputs_selective_word_removal, outputs_background_extraction)

            loss_background_extraction = criterion(outputs_background_extraction, labels) 
            loss_reconstruction = criterion(outputs_reconstruction, labels2)
            loss = loss_background_extraction + loss_reconstruction
            
            total_loss+=loss.item()*inputs.shape[0]
            total_loss_background_extraction+=loss_background_extraction.item()*inputs.shape[0]
            total_loss_reconstruction+=loss_reconstruction.item()*inputs.shape[0]

            if ((epoch+1)%100) == 0:

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
                    img_name_label = save_dir_label[condition_num] + '/label' + str(save_img_num[condition_num]) + '.png'
                    utils.save_image(inputs[k], img_name_input, normalize=True)
                    utils.save_image(outputs_background_extraction[k], img_name_outputs_background_extraction, normalize=True)
                    utils.save_image(outputs_text_extraction[k], img_name_outputs_text_extraction, normalize=True)
                    utils.save_image(outputs_selective_word_removal[k], img_name_outputs_selective_word_removal, normalize=True)
                    utils.save_image(outputs_reconstruction[k], img_name_outputs_reconstruction, normalize=True)
                    utils.save_image(labels2[k], img_name_label, normalize=True)

                    save_img_num[condition_num]+=1

    avg_loss = total_loss / len(dataset)
    avg_loss_background_extraction = total_loss_background_extraction / len(dataset)
    avg_loss_reconstruction = total_loss_reconstruction / len(dataset)

    return avg_loss, avg_loss_background_extraction, avg_loss_reconstruction


# training
def train(background_extraction, text_extraction, selective_word_removal, reconstruction, traindataloader, traindataset, criterion, optimizer, epochs, data_num, data_kind, save_dir_input_train, save_dir_outputs_background_extraction_train, 
          save_dir_outputs_text_extraction_train, save_dir_outputs_selective_word_removal_train, save_dir_outputs_reconstruction_train, save_dir_label_train,
          valdataloader, valdataset, save_dir_input_val, save_dir_outputs_background_extraction_val, save_dir_outputs_text_extraction_val, save_dir_outputs_selective_word_removal_val, save_dir_outputs_reconstruction_val, save_dir_label_val, early_stopping):


    train_loss_final_history = []
    val_loss_final_history = []
    train_loss_background_extraction_history = []
    val_loss_background_extraction_history = []
    train_loss_reconstruction_history = []
    val_loss_reconstruction_history = []
    epoch_history = []



    for epoch in (range(epochs)):

        loss_item = 0
        loss_background_extraction_item = 0
        loss_reconstruction_item = 0 
        print("Now epoch : %d/%d" %(epoch,epochs))

        if ((epoch+1)%100) == 0:
 
            save_dir_input = []
            save_dir_output_background_extraction = []
            save_dir_output_text_extraction = []
            save_dir_output_selective_word_removal = []
            save_dir_output_reconstruction = []
            save_dir_label = []

            save_img_num = []
            for i in range(data_num):
                save_img_num.append(0)

            for l in range(len(data_kind)):
                new_save_dir_input_train = save_dir_input_train + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                new_save_dir_outputs_background_extraction_train = save_dir_outputs_background_extraction_train + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                new_save_dir_outputs_text_extraction_train = save_dir_outputs_text_extraction_train + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                new_save_dir_outputs_selective_word_removal_train = save_dir_outputs_selective_word_removal_train + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                new_save_dir_outputs_reconstruction_train = save_dir_outputs_reconstruction_train + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                new_save_dir_label_train = save_dir_label_train + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                save_dir_input.append(new_save_dir_input_train)
                save_dir_output_background_extraction.append(new_save_dir_outputs_background_extraction_train)
                save_dir_output_text_extraction.append(new_save_dir_outputs_text_extraction_train)
                save_dir_output_selective_word_removal.append(new_save_dir_outputs_selective_word_removal_train)
                save_dir_output_reconstruction.append(new_save_dir_outputs_reconstruction_train)
                save_dir_label.append(new_save_dir_label_train)
                os.makedirs(new_save_dir_input_train, exist_ok=True)
                os.makedirs(new_save_dir_outputs_background_extraction_train, exist_ok=True)
                os.makedirs(new_save_dir_outputs_text_extraction_train, exist_ok=True)
                os.makedirs(new_save_dir_outputs_selective_word_removal_train, exist_ok=True)
                os.makedirs(new_save_dir_outputs_reconstruction_train, exist_ok=True)
                os.makedirs(new_save_dir_label_train, exist_ok=True)


        background_extraction.train()
        text_extraction.train()
        selective_word_removal.train()
        reconstruction.train()


        for data in tqdm(traindataloader, leave=False):
            inputs, labels, labels2, conditions = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels2 = labels2.to(device)
            conditions = conditions.to(device)

            optimizer.zero_grad()

            outputs_background_extraction = background_extraction(inputs)
            outputs_text_extraction = text_extraction(outputs_background_extraction, inputs)
            outputs_selective_word_removal = selective_word_removal(outputs_text_extraction, conditions)
            outputs_reconstruction = reconstruction(outputs_selective_word_removal, outputs_background_extraction)

            loss_background_extraction = criterion(outputs_background_extraction, labels) 
            loss_reconstruction = criterion(outputs_reconstruction, labels2)
            loss = loss_background_extraction + loss_reconstruction
            loss.backward()
            optimizer.step()

            loss_item+=loss.item()*inputs.shape[0]
            loss_background_extraction_item+=loss_background_extraction.item()*inputs.shape[0]
            loss_reconstruction_item+=loss_reconstruction.item()*inputs.shape[0]

            if ((epoch+1)%100) == 0:

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
                    img_name_label = save_dir_label[condition_num] + '/label' + str(save_img_num[condition_num]) + '.png'
                    utils.save_image(inputs[k], img_name_input, normalize=True)
                    utils.save_image(outputs_background_extraction[k], img_name_outputs_background_extraction, normalize=True)
                    utils.save_image(outputs_text_extraction[k], img_name_outputs_text_extraction, normalize=True)
                    utils.save_image(outputs_selective_word_removal[k], img_name_outputs_selective_word_removal, normalize=True)
                    utils.save_image(outputs_reconstruction[k], img_name_outputs_reconstruction, normalize=True)
                    utils.save_image(labels2[k], img_name_label, normalize=True)

                    save_img_num[condition_num]+=1

        avg_loss_train = loss_item/len(traindataset)
        avg_loss_background_extraction_train = loss_background_extraction_item/len(traindataset)
        avg_loss_reconstruction_train = loss_reconstruction_item/len(traindataset)

        train_loss_final_history.append(avg_loss_train)
        train_loss_background_extraction_history.append(avg_loss_background_extraction_train)
        train_loss_reconstruction_history.append(avg_loss_reconstruction_train)
        epoch_history.append(epoch)

        loss_val, loss_background_extraction_val, loss_reconstruction_val = validate(background_extraction, text_extraction, selective_word_removal, reconstruction, valdataloader, valdataset, criterion, epoch, data_num, data_kind, save_dir_input_val, save_dir_outputs_background_extraction_val, save_dir_outputs_text_extraction_val, save_dir_outputs_selective_word_removal_val, save_dir_outputs_reconstruction_val, save_dir_label_val)
        val_loss_final_history.append(loss_val)
        val_loss_background_extraction_history.append(loss_background_extraction_val)
        val_loss_reconstruction_history.append(loss_reconstruction_val)

        early_stopping(loss_val, background_extraction, text_extraction, selective_word_removal, reconstruction)
        if early_stopping.early_stop:         
            break


    print('Finished Training')

    return train_loss_final_history, val_loss_final_history, train_loss_background_extraction_history, val_loss_background_extraction_history, train_loss_reconstruction_history, val_loss_reconstruction_history, epoch_history, epochs



if __name__ == '__main__':
    args = get_args()

    input_root_dir = args.input
    label_root_dir =  args.label1
    label2_root_dir = args.label2
    condition_file = args.condition
    backgroung_extraction_path = args.background_extraction_path
    text_extraction_path = args.text_extraction_path
    selective_word_removal_path = args.selective_word_removal_path
    reconstruction_path = args.reconstruction_path

    data_kind = args.remove_words
    data_num = len(data_kind)

    im_list = dataloaders_finetuning.pair(label2_root_dir, condition_file)
    target_size = args.img_size
    dataset = dataloaders_finetuning.SynthTextDataset(im_list, input_root_dir, label_root_dir, label2_root_dir, target_size, data_kind, dataloaders_finetuning.resize_w_pad)

    n_train = int(len(dataset)*0.875)
    n_val = len(dataset) - n_train
    traindataset, valdataset = torch.utils.data.random_split(dataset, [n_train, n_val])

    traindataloader = DataLoader(dataset=traindataset, batch_size = args.batch, shuffle=True, num_workers=4)
    valdataloader = DataLoader(dataset=valdataset, batch_size = args.batch, shuffle=True, num_workers=4)


    background_extraction = background_extraction_module(in_channels=3)
    text_extraction = text_extraction_module(n_channels=6, n_classes=4)
    selective_word_removal = CUNET(n_layers=6, input_channels=4, filters_layer_1=64, film_type='complex', control_input_dim=data_num)
    reconstruction = reconstruction_module(n_channels=7, n_classes=3)

    background_extraction.load_state_dict(torch.load(backgroung_extraction_path))
    text_extraction.load_state_dict(torch.load(text_extraction_path))
    selective_word_removal.load_state_dict(torch.load(selective_word_removal_path))
    reconstruction.load_state_dict(torch.load(reconstruction_path))
    
    background_extraction.to(device)
    text_extraction.to(device)
    selective_word_removal.to(device)
    reconstruction.to(device)

    save_dir_input_train = './train_val_output/finetuning/train/input'
    save_dir_outputs_background_extraction_train = './train_val_output/finetuning/train/output_background_extraction'
    save_dir_outputs_text_extraction_train = './train_val_output/finetuning/train/output_text_extraction'
    save_dir_outputs_selective_word_removal_train = './train_val_output/finetuning/train/output_selective_word_removal'
    save_dir_outputs_reconstruction_train = './train_val_output/finetuning/train/output_reconstruction'
    save_dir_label_train = './train_val_output/finetuning/train/label'

    save_dir_input_val = './train_val_output/finetuning/validation/input'
    save_dir_outputs_background_extraction_val = './train_val_output/finetuning/validation/output_background_extraction'
    save_dir_outputs_text_extraction_val = './train_val_output/finetuning/validation/output_text_extraction'
    save_dir_outputs_selective_word_removal_val = './train_val_output/finetuning/validation/output_selective_word_removal'
    save_dir_outputs_reconstruction_val = './train_val_output/finetuning/validation/output_reconstruction'
    save_dir_label_val = './train_val_output/finetuning/validation/label'

    criterion = nn.MSELoss() 
    params = list(background_extraction.parameters()) + list(text_extraction.parameters()) + list(selective_word_removal.parameters()) + list(reconstruction.parameters())
    optimizer = optim.Adam(params, lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    epochs = args.epoch

    checkpoint = './train_model/finetuning'
    os.makedirs(checkpoint, exist_ok=True)
    checkpoint_name = checkpoint + '/checkpoint_model.pth'
    early_stopping = EarlyStopping(patience=args.early_stopping, verbose=True, path=checkpoint_name)

    train_loss_history, val_loss_history, train_loss_background_extraction_history, val_loss_background_extraction_history, train_loss_reconstruction_history, val_loss_reconstruction_history, epoch_history, epochs = train(background_extraction, text_extraction, selective_word_removal, reconstruction, traindataloader, traindataset, criterion, optimizer, epochs, data_num, data_kind, save_dir_input_train, save_dir_outputs_background_extraction_train, 
                                                                                                                                                                                                                                                                            save_dir_outputs_text_extraction_train, save_dir_outputs_selective_word_removal_train, save_dir_outputs_reconstruction_train, save_dir_label_train,
                                                                                                                                                                                                                                                                            valdataloader, valdataset, save_dir_input_val, save_dir_outputs_background_extraction_val, save_dir_outputs_text_extraction_val, save_dir_outputs_selective_word_removal_val, save_dir_outputs_reconstruction_val, save_dir_label_val, early_stopping)
    
    loss_train_all = []
    loss_train_all.append(train_loss_history)
    loss_train_all.append(train_loss_background_extraction_history)
    loss_train_all.append(train_loss_reconstruction_history)
    loss_val_all = []
    loss_val_all.append(val_loss_history)
    loss_val_all.append(val_loss_background_extraction_history)
    loss_val_all.append(val_loss_reconstruction_history)
    graph_name = ['final', 'background_extraction', 'reconstruction']

    for gra in range(len(loss_train_all)):
        fig1 = plt.figure()
        plt.plot(epoch_history, loss_train_all[gra], label='train')
        plt.plot(epoch_history, loss_val_all[gra], label='val')
        plt.legend()
        plt.grid()
        plt.xlabel('epoch')
        plt.title("loss")
        graph_path = './graph/finetuning'
        os.makedirs(graph_path, exist_ok=True)
        plt_name = graph_path + '/' + graph_name[gra] + '.png'
        fig1.savefig(plt_name)
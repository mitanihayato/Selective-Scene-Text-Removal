'''
    Selective Word Removal Module training code
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
    
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--epoch', type=int, default=10000, help='Number of epoch')
    parser.add_argument('--input', type=str, default='./sample_data/text_train', help='input path')
    parser.add_argument('--label', type=str, default='./sample_data/text_remove_train', help='label path')
    parser.add_argument('--condition', type=str, default='./sample_data/condition_text.csv', help='condition path')
    parser.add_argument('--remove_words', type=list, default=['China', 'France', 'Germany', 'India', 'Japan'], help='list of removal words')
    parser.add_argument('--img_size', type=int, default=512, help='Image size')
    parser.add_argument('--early_stopping', type=int, default=50, help='Early stopping epoch')

    return parser.parse_args()



# validation
def validate(net, dataloader, dataset, criterion, epoch, data_num, data_kind, save_dir_input_val, save_dir_label_val, save_dir_output_val):
    net.eval()
    with torch.no_grad():
        total_loss = 0.0

        if ((epoch+1)%50) == 0:
            save_dir_input = []
            save_dir_label = []
            save_dir_output = []

            save_img_num = []
            for i in range(data_num):
                save_img_num.append(0)

            for l in range(len(data_kind)):
                new_save_dir_input_val = save_dir_input_val + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                new_save_dir_label_val = save_dir_label_val + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                new_save_dir_output_val = save_dir_output_val + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                save_dir_input.append(new_save_dir_input_val)
                save_dir_label.append(new_save_dir_label_val)
                save_dir_output.append(new_save_dir_output_val)
                os.makedirs(new_save_dir_input_val, exist_ok=True)
                os.makedirs(new_save_dir_label_val, exist_ok=True)
                os.makedirs(new_save_dir_output_val, exist_ok=True)

        for data in dataloader:
            inputs, labels, input_condition = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            input_condition = input_condition.to(device)

            outputs = net(inputs, input_condition)
            loss = criterion(outputs, labels)
            total_loss+=loss.item()*inputs.shape[0]

            if ((epoch+1)%50) == 0:
                for k in range(int(torch.numel(input_condition)/data_num)):
                    for j in range(data_num):
                        if input_condition[k][j].item() == 1.0:
                            condition_num = j
                            break
                    img_name_input = save_dir_input[condition_num] +'/input_image' + str(save_img_num[condition_num]) + '.png'
                    img_name_label = save_dir_label[condition_num] + '/label_image' + str(save_img_num[condition_num]) + '.png'
                    img_name_outputs = save_dir_output[condition_num] + '/output_in_image' + str(save_img_num[condition_num]) + '.png'
                    utils.save_image(inputs[k], img_name_input, normalize=True)
                    utils.save_image(labels[k], img_name_label, normalize=True)
                    utils.save_image(outputs[k], img_name_outputs, normalize=True)

                    save_img_num[condition_num]+=1

    avg_loss = total_loss / len(dataset)
    return avg_loss



# training
def train(net, traindataloader, traindataset, criterion, optimizer, epochs, data_num, data_kind, save_dir_input_train, save_dir_label_train, save_dir_output_train,
          valdataloader, valdataset, save_dir_input_val, save_dir_label_val, save_dir_output_val, early_stopping):

    train_loss_history = []
    val_loss_history = []
    epoch_history = []


    for epoch in (range(epochs)):

        loss_item = 0
        print("Now epoch : %d/%d" %(epoch,epochs))

        if ((epoch+1)%50) == 0:
            save_dir_input = []
            save_dir_label = []
            save_dir_output = []

            save_img_num = []
            for i in range(data_num):
                save_img_num.append(0)

            for l in range(len(data_kind)):
                new_save_dir_input_train = save_dir_input_train + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                new_save_dir_label_train = save_dir_label_train + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                new_save_dir_output_train = save_dir_output_train + '/epoch' + str(epoch+1) + '/' + data_kind[l]
                save_dir_input.append(new_save_dir_input_train)
                save_dir_label.append(new_save_dir_label_train)
                save_dir_output.append(new_save_dir_output_train)
                os.makedirs(new_save_dir_input_train, exist_ok=True)
                os.makedirs(new_save_dir_label_train, exist_ok=True)
                os.makedirs(new_save_dir_output_train, exist_ok=True)

        net.train()

        for data in tqdm(traindataloader, leave=False):
            inputs, labels, input_condition = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            input_condition = input_condition.to(device)

            optimizer.zero_grad()
            outputs = net(inputs, input_condition)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_item+=loss.item()*inputs.shape[0]

            if ((epoch+1)%50) == 0:
                for k in range(int(torch.numel(input_condition)/data_num)):
                    for j in range(data_num):
                        if input_condition[k][j].item() == 1.0:
                            condition_num = j
                            break

                    img_name_input = save_dir_input[condition_num] +'/input_image' + str(save_img_num[condition_num]) + '.png'
                    img_name_label = save_dir_label[condition_num] + '/label_image' + str(save_img_num[condition_num]) + '.png'
                    img_name_outputs = save_dir_output[condition_num] + '/output_in_image' + str(save_img_num[condition_num]) + '.png'
                    utils.save_image(inputs[k], img_name_input, normalize=True)
                    utils.save_image(labels[k], img_name_label, normalize=True)
                    utils.save_image(outputs[k], img_name_outputs, normalize=True)

                    save_img_num[condition_num]+=1

        avg_loss_train = loss_item/len(traindataset)
        train_loss_history.append(avg_loss_train)
        epoch_history.append(epoch)

        loss_val = validate(net, valdataloader, valdataset, criterion, epoch, data_num, data_kind, save_dir_input_val, save_dir_label_val, save_dir_output_val)
        val_loss_history.append(loss_val)

        early_stopping(loss_val, net)
        if early_stopping.early_stop:         
            break

    print('Finished Training')

    return train_loss_history, val_loss_history, epoch_history, epochs



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


    n_train = int(len(dataset)*0.875)
    n_val = len(dataset) - n_train
    traindataset, valdataset = torch.utils.data.random_split(dataset, [n_train, n_val])

    traindataloader = DataLoader(dataset=traindataset, batch_size = args.batch, shuffle=True, num_workers=4)
    valdataloader = DataLoader(dataset=valdataset, batch_size = args.batch, shuffle=True, num_workers=4)

    net = CUNET(n_layers=6, input_channels=4, filters_layer_1=64, film_type='complex', control_input_dim=data_num)
    net.to(device)

    save_dir_input_train = './train_val_output/selective_word_removal_module/train/input'
    save_dir_label_train = './train_val_output/selective_word_removal_module/train/label'
    save_dir_output_train = './train_val_output/selective_word_removal_module/train/output'

    save_dir_input_val = './train_val_output/selective_word_removal_module/validation/input'
    save_dir_label_val = './train_val_output/selective_word_removal_module/validation/label'
    save_dir_output_val = './train_val_output/selective_word_removal_module/validation/output'

    criterion = nn.MSELoss() 
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    epochs = args.epoch

    checkpoint = './train_model/selective_word_removal_module'
    os.makedirs(checkpoint, exist_ok=True)
    checkpoint_name = checkpoint + '/checkpoint_model.pth'
    early_stopping = EarlyStopping(patience=args.early_stopping, verbose=True, path=checkpoint_name)

    train_loss_history, val_loss_history, epoch_history, epochs = train(net, traindataloader, traindataset, criterion, optimizer, epochs, data_num, data_kind,
                                                                        save_dir_input_train, save_dir_label_train, save_dir_output_train,
                                                                        valdataloader, valdataset, save_dir_input_val, save_dir_label_val, save_dir_output_val, early_stopping)
    
    fig1 = plt.figure()
    plt.plot(epoch_history, train_loss_history, label='train')
    plt.plot(epoch_history, val_loss_history, label='val')
    plt.legend()
    plt.grid()
    plt.xlabel('epoch')
    plt.title("loss")
    graph_path = './graph/selective_word_removal_module.png'
    os.makedirs('./graph', exist_ok=True)
    fig1.savefig(graph_path)
'''
    Text Extraction Module training code
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
from tools.pytorchtools import EarlyStopping
import re
from tools import dataloaders_text_extraction
from models.Text_extraction_module.text_extra_module import text_extraction_module
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
    
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--epoch', type=int, default=10000, help='Number of epoch')
    parser.add_argument('--input1', type=str, default='./sample_data/background_train', help='input(background) path')
    parser.add_argument('--input2', type=str, default='./sample_data/scene_train', help='input(scene) path')
    parser.add_argument('--label', type=str, default='./sample_data/text_train', help='label path')
    parser.add_argument('--img_size', type=int, default=512, help='Image size')
    parser.add_argument('--early_stopping', type=int, default=50, help='Early stopping epoch')

    return parser.parse_args()




# validation
def validate(net, dataloader, dataset, criterion, epoch, save_dir_input, save_dir_input2, save_dir_label, save_dir_output):
    net.eval()
    with torch.no_grad():
        total_loss = 0.0
        i = 0

        if ((epoch+1)%100) == 0:
            new_save_dir_input = save_dir_input + '/epoch' + str(epoch+1)
            new_save_dir_input2 = save_dir_input2 + '/epoch' + str(epoch+1)
            new_save_dir_label = save_dir_label + '/epoch' + str(epoch+1)
            new_save_dir_output = save_dir_output + '/epoch' + str(epoch+1)
            os.makedirs(new_save_dir_input, exist_ok=True)
            os.makedirs(new_save_dir_input2, exist_ok=True)
            os.makedirs(new_save_dir_label, exist_ok=True)
            os.makedirs(new_save_dir_output, exist_ok=True)

        for data in dataloader:
            inputs, inputs2, labels = data
            inputs = inputs.to(device)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)

            outputs = net(inputs, inputs2)
            loss = criterion(outputs, labels)
            total_loss+=loss.item()*inputs.shape[0]


            if ((epoch+1)%100) == 0:
                img_name_input = new_save_dir_input +'/input_back_image' + str(i) + '.png'
                img_name_input2 = new_save_dir_input2 +'/input_scene_image' + str(i) + '.png'
                img_name_label = new_save_dir_label + '/label_text_image' + str(i) + '.png'
                img_name_output = new_save_dir_output + '/output_image' + str(i) + '.png'
                utils.save_image(inputs, img_name_input, normalize=True)
                utils.save_image(inputs2, img_name_input2, normalize=True)
                utils.save_image(labels, img_name_label, normalize=True)
                utils.save_image(outputs, img_name_output, normalize=True)
                i+=1 

    avg_loss = total_loss / len(dataset)
    return avg_loss



# training
def train(net, traindataloader, traindataset, criterion, optimizer, epochs, save_dir_input_train, save_dir_input2_train, save_dir_label_train, save_dir_output_train,
          valdataloader, valdataset, save_dir_input_val, save_dir_input2_val, save_dir_label_val, save_dir_output_val, early_stopping):

    train_loss_history = []
    val_loss_history = []
    epoch_history = []

    i = 0

    for epoch in (range(epochs)):

        loss_item = 0
        print("Now epoch : %d/%d" %(epoch,epochs))

        if ((epoch+1)%100) == 0:
            new_save_dir_input_train = save_dir_input_train + '/epoch' + str(epoch+1)
            new_save_dir_input2_train = save_dir_input2_train + '/epoch' + str(epoch+1)
            new_save_dir_label_train = save_dir_label_train + '/epoch' + str(epoch+1)
            new_save_dir_output_train = save_dir_output_train + '/epoch' + str(epoch+1)
            os.makedirs(new_save_dir_input_train, exist_ok=True)
            os.makedirs(new_save_dir_input2_train, exist_ok=True)
            os.makedirs(new_save_dir_label_train, exist_ok=True)
            os.makedirs(new_save_dir_output_train, exist_ok=True)


        net.train()

        for data in tqdm(traindataloader, leave=False):
            inputs, inputs2, labels = data
            inputs = inputs.to(device)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs, inputs2)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_item+=loss.item()*inputs.shape[0]

            if ((epoch+1)%100) == 0:
                img_name_input = new_save_dir_input_train +'/input_image' + str(i) + '.png'
                img_name_input2 = new_save_dir_input2_train +'/input_image' + str(i) + '.png'
                img_name_label = new_save_dir_label_train + '/label_image' + str(i) + '.png'
                img_name_outputs = new_save_dir_output_train + '/output_in_image' + str(i) + '.png'

                utils.save_image(inputs, img_name_input, normalize=True)
                utils.save_image(inputs2, img_name_input2, normalize=True)
                utils.save_image(labels, img_name_label, normalize=True)
                utils.save_image(outputs, img_name_outputs, normalize=True)

                i+=1

        avg_loss_train = loss_item/len(traindataset)
        train_loss_history.append(avg_loss_train)
        epoch_history.append(epoch)

        loss_val = validate(net, valdataloader, valdataset, criterion, epoch, save_dir_input_val, save_dir_input2_val, save_dir_label_val, save_dir_output_val)
        val_loss_history.append(loss_val)

        early_stopping(loss_val, net)
        if early_stopping.early_stop: 
            break

    print('Finished Training')

    return train_loss_history, val_loss_history, epoch_history, epochs



if __name__ == '__main__':
    args = get_args()

    input_root_dir = args.input1
    input2_root_dir = args.input2
    label_root_dir = args.label

    im_list = dataloaders_text_extraction.pair(input_root_dir)
    target_size = args.img_size
    dataset = dataloaders_text_extraction.SynthTextDataset(im_list, input_root_dir, input2_root_dir, label_root_dir, target_size, dataloaders_text_extraction.resize_w_pad)

    n_train = int(len(dataset)*0.875)
    n_val = len(dataset) - n_train
    traindataset, valdataset = torch.utils.data.random_split(dataset, [n_train, n_val])

    traindataloader = DataLoader(dataset=traindataset, batch_size = args.batch, shuffle=True, num_workers=4)
    valdataloader = DataLoader(dataset=valdataset, batch_size = args.batch, shuffle=True, num_workers=4)

    net = text_extraction_module(n_channels=6, n_classes=4)
    net.to(device)

    save_dir_input_train = './train_val_output/text_extraction_module/train/input'
    save_dir_input2_train = './train_val_output/text_extraction_module/train/input2'
    save_dir_label_train = './train_val_output/text_extraction_module/train/label'
    save_dir_output_train = './train_val_output/text_extraction_module/train/output'

    save_dir_input_val = './train_val_output/text_extraction_module/validation/input'
    save_dir_input2_val = './train_val_output/text_extraction_module/validation/input2'
    save_dir_label_val = './train_val_output/text_extraction_module/validation/label'
    save_dir_output_val = './train_val_output/text_extraction_module/validation/output'

    criterion = nn.MSELoss() 
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    epochs = args.epoch

    checkpoint = './train_model/text_extraction_module'
    os.makedirs(checkpoint, exist_ok=True)
    checkpoint_name = checkpoint + '/checkpoint_model.pth'
    early_stopping = EarlyStopping(patience=args.early_stopping, verbose=True, path=checkpoint_name)

    train_loss_history, val_loss_history, epoch_history, epochs = train(net, traindataloader, traindataset, criterion, optimizer, epochs, 
                                                                        save_dir_input_train, save_dir_input2_train, save_dir_label_train, save_dir_output_train,
                                                                        valdataloader, valdataset, save_dir_input_val, save_dir_input2_val, save_dir_label_val, save_dir_output_val, early_stopping)
    

    fig1 = plt.figure()
    plt.plot(epoch_history, train_loss_history, label='train')
    plt.plot(epoch_history, val_loss_history, label='val')
    plt.legend()
    plt.grid()
    plt.xlabel('epoch')
    plt.title("loss")
    graph_path = './graph/text_extraction_module.png'
    os.makedirs('./graph', exist_ok=True)
    fig1.savefig(graph_path)
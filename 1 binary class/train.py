## IMPORTS 
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
import os
import numpy as np
import tqdm
import datetime
import math
from validation import val_multi
import glob
from load_dataset import Load_Dataset
from tensorboardX import SummaryWriter
import torch.nn as nn
from loss import *
from RAUNet import RAUNet

# SETTING THE DEVICE ID TO 0 FOR GPU USAGE
device_ids = [0]

## SETTING VARIABLES TO USE ACCROSS THE CODE
parse=argparse.ArgumentParser()
num_classes=1
lra=0.0001
batch_size = 16

## A FUNCTION TO CHANGE THE LEARNING RATE
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lra* (0.8 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

## LOADING THE FILE NAME FUNCTION
def load_filename():
    train_file_names = glob.glob('C:/Users/asemi/Desktop/Data/Sunnybrook/2 classes/Augmented/Training/Images/*.png')
    val_file_names = glob.glob('C:/Users/asemi/Desktop/Data/Sunnybrook/2 classes/Augmented/Validation/Images/*.png')
    return train_file_names,val_file_names


## Main FUNCTION 
def main():

    # Initializing the Model and setting the parameters
    mod = RAUNet(num_classes=num_classes,pretrained=True)

    ## Setting the model to train on the GPU
    model = mod.cuda(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids)

    ## Intializing the Loss Function and Optimizer 
    criterion =DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lra)
    
    
    ## LOADING the DATASET 
    train_file, val_file = load_filename()
    train_dataset = Load_Dataset(train_file)
    val_dataset= Load_Dataset(val_file)

    # LOADERS to be used in training and Validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True) # drop_last=True
    val_load=DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1,drop_last=True)
    
    # Calling the train model Function
    train_model(model, criterion, optimizer, train_loader, val_load, num_classes)


## Training Model Function 
def train_model(model, criterion, optimizer, dataload,val_load,num_classes,num_epochs=50):

    # Intializing Lists to store the loss of each epoch and save them 
    loss_list=[]
    dice_list=[]
    
    # Path to write the Logs
    logs_dir = 'Logs/T{}/'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.mkdir(logs_dir)

    # Initializing Writer 
    writer = SummaryWriter(logs_dir)

    # Training each epoch loop
    for epoch in range(num_epochs):
        
        # Specifiy that the model is in training mode
        model.train()


        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        # tqdm to print the progress of each epoch and the loss 
        dt_size = len(dataload.dataset)
        tq = tqdm.tqdm(total=math.ceil(dt_size/batch_size))
        tq.set_description('Epoch {}'.format(epoch))
        epoch_loss =[]
        step = 0

        # Loop on the loader to get batches
        for x, y in dataload:
            step += 1

            # Make the inputs and labels work on GPU
            inputs = x.cuda(device_ids[0])
            # y=y.long()
            labels = y.cuda(device_ids[0])

            # Making the grads zero for each epoch to prevent summation           
            optimizer.zero_grad()

            # Prediciton of the model
            outputs = model(inputs)
            # Calculation Loss 
            loss = criterion(outputs, labels)
            # Backward propagation
            loss.backward()
            # Optimizing the model
            optimizer.step()
            # Printing some info
            tq.update(1)
            epoch_loss.append(loss.item())
            epoch_loss_mean = np.mean(epoch_loss).astype(np.float64)
            tq.set_postfix(loss='{0:.3f}'.format(epoch_loss_mean))
        loss_list.append(epoch_loss_mean)
        tq.close()
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss_mean))
        # Validation Function and getting dice
        dice, iou =val_multi(model, criterion, val_load, num_classes,batch_size,device_ids)
        writer.add_scalar('Loss', epoch_loss_mean, epoch)
        writer.add_scalar('Dice', dice, epoch)
        writer.add_scalar('IoU', iou, epoch)
        dice_list.append([dice,iou])
        adjust_learning_rate(optimizer, epoch)
        # Saving Weights
        torch.save(model.module.state_dict(), logs_dir + 'weights_{}.pth'.format(epoch))
        fileObject = open(logs_dir+'LossList.txt', 'w')
        for ip in loss_list:
            fileObject.write(str(ip))
            fileObject.write('\n')
        fileObject.close()
        fileObject = open(logs_dir + 'dice_list.txt', 'w')
        for ip in dice_list:
            fileObject.write(str(ip))
            fileObject.write('\n')
        fileObject.close()

    writer.close()
    return model


if __name__ == '__main__':

    main()

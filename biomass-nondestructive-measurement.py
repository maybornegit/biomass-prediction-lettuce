#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:19:55 2024

@author: morganmayborne
"""

##### Non-Destructive Biomass Measurement Training

import torch, random, tqdm
import torch.nn as nn
from torch.utils.data import Dataset
from os import listdir
import pandas as pd
from PIL import Image, ImageEnhance
import numpy as np


class LettuceDataset(Dataset):
    def __init__(self, rgbd_set, transform=None):
        self.transform = transform
        self.dataset_rgbd = rgbd_set

    def __len__(self):
        return len(self.dataset_rgbd)

    def __getitem__(self, index):
        data_info = self.dataset_rgbd[index]
        rgbd_image = data_info[0].copy()
        rgbd_image = np.transpose(rgbd_image, (2, 0, 1))  # Assuming rgbd_image is [height, width, channels]
        rgbd = torch.from_numpy(rgbd_image).float()  # Convert to PyTorch tensor
        true_biomass = data_info[1]
        return rgbd, true_biomass
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=5, stride=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(512 * 1 * 1, 1)  # Adjusted for regression output size

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.relu5(self.bn5(self.conv5(x)))
        # print(x.shape)
        x = x.view(-1, 512 * 1 *1)  # Flatten the tensor
        x = self.dropout(x)
        x = self.fc(x)
        # print(x.shape)
        return torch.squeeze(x)
    
def augmentation(rgbd_image):
    cur_set = []
    
    # Rotations
    ninety = np.rot90(rgbd_image.copy())
    cur_set.append(ninety)
    
    oneeighty = np.rot90(ninety.copy())
    cur_set.append(oneeighty)
    
    twoseventy = np.rot90(oneeighty.copy())
    cur_set.append(twoseventy)
    
    # Flips
    flip_hor = np.fliplr(rgbd_image.copy())
    cur_set.append(flip_hor)
    
    flip_vert = np.flipud(rgbd_image.copy())
    cur_set.append(flip_vert)
    
    # Brightness changes
    brightness_factors = [0.8, 0.9, 1.1, 1.2]
    for img in cur_set[:5]:  # Only apply brightness to original and rotations
        for factor in brightness_factors:
            augmented_img = img.copy().astype(np.uint8)
            rgb_part = augmented_img[:, :, :3]
            im = Image.fromarray(rgb_part)
            enhancer = ImageEnhance.Brightness(im)
            im = enhancer.enhance(factor)
            augmented_img[:, :, :3] = np.array(im)
            cur_set.append(augmented_img)
    
    cur_set.append(rgbd_image)
    cur_set = [np.ascontiguousarray(arr) for arr in cur_set]
    return cur_set

### Important Database Directories
coreDir = "./Field measurements and code/"
trainDir = "Training and validation"
trueDir = "Field measurements 2.csv"
depthDir = 'Depth image dataset/'

### Set-up for Image Locations + True Biomass
trainPaths = [f for f in listdir(coreDir+trainDir)]
trueData = pd.read_csv(coreDir+trueDir)
trainPathswTrue = [(coreDir+trainDir+'/'+trainPaths[i], float(trueData[trueData['image']==trainPaths[i]]['LFW'].iloc[0])) for i in range(len(trainPaths))]
    

## Set-up coupling with depth images and create 4-depth RGB-D images    
types = ['Flandria', 'Tiberius', 'Locarno']
for i in range(len(trainPaths)):
    if trainPaths[i][0] == '1':
        trainPathswTrue[i] = (trainPathswTrue[i][0], trainPathswTrue[i][1], coreDir+depthDir+types[0]+'/'+'Depth-'+trainPaths[i][:-4]+'.png')
    elif trainPaths[i][0] == '2':
        trainPathswTrue[i] = (trainPathswTrue[i][0], trainPathswTrue[i][1], coreDir+depthDir+types[1]+'/'+'Depth-'+trainPaths[i][:-4]+'.png')
    elif trainPaths[i][0] == '3':
        trainPathswTrue[i] = (trainPathswTrue[i][0], trainPathswTrue[i][1], coreDir+depthDir+types[2]+'/'+'Depth-'+trainPaths[i][:-4]+'.png')
     
rgbdTrue = []
size_ = 128
for i in range(len(trainPaths)):
    try:
        rgb = np.asarray(Image.open(trainPathswTrue[i][0]).resize((size_,size_)))
        depth = np.asarray(Image.open(trainPathswTrue[i][2]))
        y,x = depth.shape
        startx = x//2-(size_//2)
        starty = y//2-(size_//2)    
        depth = depth[starty:starty+size_,startx:startx+size_]
        depth = Image.fromarray(depth)
        depth = np.asarray(depth).reshape((size_,size_,-1))
        # print(rgb.shape, depth.shape)
        rgbd = np.concatenate([rgb, depth], axis=2)
        rgbdTrue.append((rgbd,trainPathswTrue[i][1]))
    except Exception as error:
        print(error)
        
augmentSet = []
for k in range(len(rgbdTrue)):
    augment = augmentation(rgbdTrue[k][0])
    [augmentSet.append((augment[x],rgbdTrue[k][1])) for x in range(len(augment))]
           
training = list(range(len(augmentSet)))
random.shuffle(training)
train = training[:int(.8*len(augmentSet))]
valid = training[int(.8*len(augmentSet)):]

train_dataset = LettuceDataset([augmentSet[idx] for idx in train])  
valid_dataset = LettuceDataset([augmentSet[idx] for idx in valid])
print(len(train_dataset), len(valid_dataset))
# print(dataset[0][0].shape, dataset[0][1])

print("\nBegin test of training code\n")
  
torch.manual_seed(1)
np.random.seed(1)
train_file = "./Biomass_results.txt"

bat_size = 128
max_epochs = 500
train_ldr = torch.utils.data.DataLoader(train_dataset,
  batch_size=bat_size, shuffle=True)
device = torch.device("cpu")

net = CNN().to(device)
net.train()  # set mode

lrn_rate = 1e-2
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),
  lr=lrn_rate)

milestones = np.arange(20,max_epochs,20)  # Drop learning rate by a factor of 0.1 at these milestones
gamma = 0.1  # Drop factor
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

for epoch in range(0, 500):
    print(epoch)
    # T.manual_seed(1 + epoch)  # recovery reproduce
    epoch_loss = 0.0  # sum avg loss per item
      
    for (batch_idx, batch) in enumerate(tqdm.tqdm(train_ldr)):
      X = batch[0].float()  # predictors shape [10,8]
      Y = batch[1].float()  # targets shape [10,1] 
      
      optimizer.zero_grad()
      oupt = net(X)            # shape [10,1]
      
      loss_val = loss_func(oupt, Y)  # avg loss in batch
      epoch_loss += loss_val.item()  # a sum of averages
      loss_val.backward()
      optimizer.step()
      
    print(epoch, epoch_loss)
    if epoch % 100 == 0:
      print(" epoch = %4d   loss = %0.4f" % \
        (epoch, epoch_loss))
      # TODO: save checkpoint

print("\nDone ")
## torch DataLoader
## CNN Functions
## Figure out the loss function
## Training Process





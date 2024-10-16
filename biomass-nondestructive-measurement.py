# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:19:55 2024

@author: morganmayborne
"""
import sys

import torch, random, tqdm, json, os, time, sys, cv2
import torch.nn as nn
from os import listdir
import pandas as pd
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from biomass_cnn_models import CNN_128, CNN_640, CNN_Delta, MAPE
from augment_utils import data_augmentations, LettuceDataset
import data_utils


def close_event():
    plt.close()  # timer calls this function after 3 seconds and closes the window


plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle')

args = sys.argv[1:]
if '--name' in args:
    idx = args.index('--name')
    TRIAL_NAME = str(args[idx + 1])
else:
    TRIAL_NAME = 'testing'
if '-e' in args:
    idx = args.index('-e')
    max_epochs = int(args[idx + 1])
else:
    max_epochs = 150
if '-b' in args:
    idx = args.index('-b')
    bat_size = int(args[idx + 1])
else:
    bat_size = 16
if '-lr' in args:
    idx = args.index('-lr')
    lrn_rate = float(args[idx + 1])
else:
    lrn_rate = 3e-5
if '--seg' in args:
    idx = args.index('--seg')
    segmentation = args[idx + 1] == 'True' or args[idx+1] == '1'
else:
    segmentation = False
if '--zoom' in args:
    idx = args.index('--zoom')
    artifZoom = args[idx + 1] == 'True' or args[idx+1] == '1'
else:
    artifZoom = False
if '--plot' in args:
    idx = args.index('--plot')
    plot_results = args[idx + 1] == 'True' or args[idx+1] == '1'
else:
    plot_results = True
if '-d' in args:
    idx = args.index('-d')
    test_type = str(args[idx + 1])
else:
    test_type = 'my_data_delta'
if '--augflip' in args:
    idx = args.index('--augflip')
    aug_flip = args[idx + 1] == 'True' or args[idx+1] == '1'
else:
    aug_flip = True
if '--augrot' in args:
    idx = args.index('--augrot')
    aug_rot = args[idx + 1] == 'True' or args[idx+1] == '1'
else:
    aug_rot = True
if '--augbright' in args:
    idx = args.index('--augbright')
    aug_bright = int(args[idx+1])
else:
    aug_bright = 2
if '--deltajump' in args:
    idx = args.index('--deltajump')
    delta_jump = int(args[idx+1])
else:
    delta_jump = 3
if '-m' in args:
    idx = args.index('-m')
    max_pics = int(args[idx+1])
else:
    if test_type == 'my_data_delta':
        max_pics = 800
    else:
        max_pics = 2500
if '--loss' in args:
    idx = args.index('--loss')
    loss = str(args[idx+1])
else:
    loss = 'MAPE'

presets = ['Name: ', TRIAL_NAME,' (--name)',
           'Epochs: ', max_epochs,' (-e)',
           'Batch Size: ', bat_size,' (-b)',
           'Learning Rate: ', lrn_rate,' (-lr)',
           'Segmentation: ', segmentation,' (--seg)',
           'Artifical Zoom: ', artifZoom,' (--zoom)',
           'Plot Results: ', plot_results,' (--plot)',
           'Training Data: ', test_type, ' (-d)',
           'Flip Augment: ',aug_flip,' (--augflip)',
           'Rotate Augment: ',aug_rot,' (--augrot)',
           'Brightness Augment: ',aug_bright,' (--augbright)',
           'Delta Jump: ',delta_jump,' (--deltajump)',
           'Max Pictures: ',max_pics,' (-m)',
           'Loss: ',loss,' (--loss)'
           ]

for i in range(len(presets) // 3):
    print(presets[3 * i] + str(presets[3* i + 1])+presets[3 * i+2])

#print('Additional Presets:','Flip Augment (--augflip)',aug_flip,'Rotate Augment (--augrot)',aug_rot,'Brightness Augment (--augbright)',aug_bright,'Delta Jump (--deltajump)',delta_jump,'Max Pictures (-m)',max_pics)

confirmation = input('\nAre these presets okay? (y/n) ')
if confirmation not in ['y', 'Y', 'yes', 'Yes', 'YES', 't', 'u', 'h', '6']:
    exit()

train_split = 0.8
train_file = '../old_cnn_results/' + TRIAL_NAME + '.pt'
losses = []
banned_for_test = ['2024-09-17-14', '2024-09-20-7','2024-09-10-0', '2024-09-11-0', '2024-09-12-0', '2024-09-13-0', '2024-09-16-0', '2024-09-17-0', '2024-09-18-0', '2024-09-19-0', '2024-09-20-0','2024-09-10-3', '2024-09-11-3', '2024-09-12-3', '2024-09-13-3', '2024-09-16-3', '2024-09-17-3', '2024-09-18-3', '2024-09-19-7', '2024-09-20-7','2024-09-10-7', '2024-09-11-7', '2024-09-12-7', '2024-09-13-7', '2024-09-16-7', '2024-09-17-7', '2024-09-18-7', '2024-09-19-11', '2024-09-20-11', '2023-09-24-4', '2024-09-25-3', '2024-09-26-3', '2024-09-27-3']
#banned_for_test = ['2024-09-17-14', '2024-09-20-7']
torch.manual_seed(1)
np.random.seed(1)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the CPU")

print('\n-----Dataset Loading-----\n')
###
aug_factor = (aug_bright+1)*(3*aug_rot+2*aug_flip)+1
if test_type == 'xu_data':
    ### Important Database Directories
    coreDir = "../Field measurements and code/"
    trainDir = "Training and validation"
    trueDir = "Field measurements 2.csv"
    depthDir = 'Depth image dataset/'

    ### Set-up for Image Locations + True Biomass
    rgbdTrue = data_utils.test_data_128(coreDir, trainDir, trueDir, depthDir)
    net = CNN_128().to(device)
elif test_type == 'third_auton_data':
    ### Important Database Directories
    coreDir = "../AutonomousGreenhouseChallengeData/"
    trainDir = "/RGBImages"
    trueDir = "/GroundTruth/GroundTruth_All_388_Images.json"
    depthDir = '/DepthImages'

    ### Set-up for Image Locations + True Biomass
    rgbdTrue = data_utils.test_data_third_auton(coreDir, trainDir, trueDir, depthDir)
    net = CNN_640().to(device)
elif test_type == 'my_data_standard':
    ### Important Database Directories
    coreDir = "../CustomI2GROW_Dataset"
    trainDir = "/RGBDImages"
    trueDir = "/Biomass_Info_Ground_Truth.csv"

    ### Set-up for Image Locations + True Biomass
    rgbdTrue = data_utils.test_data_my_data(coreDir, trainDir, trueDir, segmentation, artifZoom,banned_for_test,aug_factor, max_pics)
    net = CNN_640().to(device)
elif test_type == 'my_data_delta':
    coreDir = "../CustomI2GROW_Dataset"
    trainDir = "/RGBDImages/"
    trueDir = "/Biomass_Info_Ground_Truth.csv"

    suc_sheet = coreDir+ '/successor_sheet.csv'

    rgbdTrue = data_utils.test_data_my_data_delta(coreDir, trainDir, trueDir, suc_sheet, segmentation, artifZoom,banned_for_test, delta_jump,aug_factor, max_pics)
    net = CNN_Delta().to(device)
else:
    raise NotImplementedError

print('\n-----Dataset Loading Complete-----\n')

print('\n-----Augmentation-----\n')

# Create training and validation datasets with augmentations
augmentSet_train, augmentSet_valid = data_augmentations(rgbdTrue, train_split, aug_flip,aug_rot, aug_bright)
print(len(augmentSet_train))

train_dataset = LettuceDataset(augmentSet_train)
valid_dataset = LettuceDataset(augmentSet_valid)

print('\n-----Augmentation Complete-----\n')

print('\n-----Training Initialized-----\n')

##### Data Loading
train_ldr = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=bat_size, shuffle=True)

valid_ldr = torch.utils.data.DataLoader(valid_dataset,
                                        batch_size=bat_size, shuffle=False)

##### Number of Parameters (if nec.)
# print('Total Parameters: ', sum(p.numel() for p in net.parameters()))

##### LOSS FUNCTION
if loss == 'MSE':
    loss_func = torch.nn.MSELoss()
    mse_calc = torch.nn.MSELoss()
elif loss == 'MAPE':
    loss_func = MAPE()
    mse_calc = torch.nn.MSELoss()
else:
    raise NotImplementedError
optimizer = torch.optim.Adam(net.parameters(),lr=lrn_rate)

##### SCHEDULED DROPS (if nec.)
milestones = np.arange(20,100,20)  # Drop learning rate by a factor of 0.75 at these milestones
gamma = 0.5  # Drop factor
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

train_loss = []
valid_loss = []
for epoch in range(0, max_epochs):
    print('-----Iteration' + str(epoch) + '-----')
    # T.manual_seed(1 + epoch)  # recovery reproduce
    net.train()
    epoch_loss = 0.0  # sum avg loss per item
    sum_true = 0.0
    ##### TRAINING LOOP
    for (batch_idx, batch) in enumerate(tqdm.tqdm(train_ldr)):
        X, Y = batch[0].float(), batch[1].float()
        sum_true += torch.sum(Y).detach()
        optimizer.zero_grad()
        oupt = net(X).reshape((-1,))  # shape [10,1]
        len_batch = oupt.shape[0]
        loss_val = loss_func(Y,oupt)  # avg loss in batch
        loss_val.backward()
        optimizer.step()

        epoch_loss += mse_calc(oupt, Y).detach() * len_batch  # a sum of averages

    ##### TRAINING EVALUATION
    mse = epoch_loss / len(train_dataset)
    mean_error = np.sqrt(mse)
    nrmse = mean_error / (sum_true / len(train_dataset))
    train_loss.append(nrmse)

    valid_loss_iter = 0.0  # sum avg loss per item
    sum_true = 0.0
    net.eval()
    ##### VALIDATION LOOP
    for (batch_idx, batch) in enumerate(tqdm.tqdm(valid_ldr)):
        X = batch[0].float()  # predictors shape [10,8]
        Y = batch[1].float()  # targets shape [10,1]
        sum_true += torch.sum(Y)
        oupt = net(X).detach().reshape((-1,))  # shape [10,1]
        len_batch = oupt.shape[0]
        loss_val = mse_calc(oupt, Y).detach() * len_batch  # avg loss in batch
        valid_loss_iter += loss_val.item()  # a sum of averages

    ##### VALIDATION EVALUATION
    mse = valid_loss_iter / len(valid_dataset)
    mean_error = np.sqrt(mse)
    nrmse = mean_error / (sum_true / len(valid_dataset))
    valid_loss.append(nrmse)

    print('\nCurrent Validation NRMSE Loss:' + str(valid_loss[-1]))

    ##### PLOT (if necessary every epoch)
    '''if epoch > 0:
        fig = plt.figure(1)
        timer = fig.canvas.new_timer(
            interval=3000)  # creating a timer object and setting an interval of 3000 milliseconds
        timer.add_callback(close_event)
        plt.plot(np.arange(1, len(train_loss) + 1), train_loss, label='Training Loss')
        plt.plot(np.arange(1, len(train_loss) + 1), valid_loss, label='Validation Loss')
        plt.legend()
        plt.title('Training Loss vs Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('NRMSE Loss per Image')
        timer.start()
        plt.show()'''

print('\n-----Training Complete-----\n')

print('\n-----Trained Weights Saved-----\n')

torch.save(net.state_dict(), train_file)

print('\n-----Trained Weights Saved-----\n')

print('\n-----Final Analysis-----\n')

epoch_loss = 0.0  # sum avg loss per item
train_loss = np.array(train_loss)
valid_loss = np.array(valid_loss)

print('Train Loss: ', train_loss[-1], 'Validation Loss: ', valid_loss[-1])
losses.append([train_loss[-1], valid_loss[-1]])

valid_ldr = torch.utils.data.DataLoader(valid_dataset,
                                        batch_size=1, shuffle=False)

actual_plot = []
predicted_plot = []
nrmse_loss = []
for (batch_idx, batch) in enumerate(tqdm.tqdm(valid_ldr)):
    X = batch[0].float()  # predictors shape [10,8]
    Y = batch[1].float()  # targets shape [10,1]

    oupt = net(X).detach().reshape((-1,))  # shape [10,1]
    for i in range(X.shape[0]):
        actual_plot.append(Y[i])
        # print(Y[i])
        predicted_plot.append(oupt[i])

    for i in range(Y.shape[0]):
        loss_val = mse_calc(oupt[i], Y[i]).detach()
        rmse = np.sqrt(loss_val.item())
        nrmse = rmse / Y[i]
        nrmse_loss.append((Y[i], nrmse))

nrmse_error_chart = []
for i in range(len(nrmse_loss)):
    if i == 0:
        nrmse_error_chart.append(list(nrmse_loss[i]))
    elif nrmse_loss[i][0] in [nrmse_error_chart[key][0] for key in range(len(nrmse_error_chart))]:
        idx = [nrmse_error_chart[key][0] for key in range(len(nrmse_error_chart))].index(nrmse_loss[i][0])
        nrmse_error_chart[idx].append(nrmse_loss[i][1])
    else:
        nrmse_error_chart.append(list(nrmse_loss[i]))

nrmse_error_chart = [(nrmse_error_chart[i][0], sum(nrmse_error_chart[i][1:]) / len(nrmse_error_chart[i][1:])) for i in
                     range(len(nrmse_error_chart))]
weights = [data[0] for data in nrmse_error_chart]
error = [data[1] for data in nrmse_error_chart]

actual_plot = np.array(actual_plot)
predicted_plot = np.array(predicted_plot)

if plot_results:
    ##### LOSS CURVES (TRAINING + VALIDATION)
    plt.figure(1)
    plt.plot(np.arange(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(np.arange(1, len(train_loss) + 1), valid_loss, label='Validation Loss')
    plt.legend()
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('NRMSE Loss per Image')

    ##### ACTUAL V. PREDICTION VISUALIZATION
    plt.figure(2)
    plt.scatter(actual_plot, predicted_plot, label='Training Loss')
    plt.plot(np.arange(0, np.max(actual_plot)), np.arange(0, np.max(actual_plot)), 'k-')
    plt.legend()
    plt.title('Actual v. Predicted Scatter Plot')
    plt.xlabel('Actual Biomass')
    plt.ylabel('Predicted Biomass')

    ##### NRMSE VARIANCE BY WEIGHT
    plt.figure(3)
    plt.bar(weights, error, label='Error')
    plt.legend()
    plt.title('Error v. Weight')
    plt.xlabel('Actual Biomass')
    plt.ylabel('NRMSE Error Metric')

##### FINAL METRICS (TRAIN LOSS, VALID LOSS)
print('Train Loss, Validation Loss', losses)
plt.show()

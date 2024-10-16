import tqdm, json, cv2, os
import numpy as np
import pandas as pd
from datetime import datetime
from os import listdir
from PIL import Image
import random

def isolate_test_data(data_labels,isolate_data, CNN_type):
    data_filtered = []
    if CNN_type == 'delta':
        for i in range(len(data_labels)):
            if (data_labels[i][0] not in isolate_data and data_labels[i][1] not in isolate_data):
                # print(trainPaths[i])
                data_filtered.append(data_labels[i])
    elif CNN_type == 'standard':
        for i in range(len(data_labels)):
            if (data_labels[i] not in isolate_data):
                # print(trainPaths[i])
                data_filtered.append(data_labels[i])
    return data_filtered
def padding_image(image, target_size):
    """
    Pads an image to the target size with equal padding above and below.

    Parameters:
    - image (np.ndarray): The input image to be padded. Shape should be (height, width, channels).
    - target_size (tuple): The target size as (target_height, target_width).

    Returns:
    - np.ndarray: The padded image.
    """
    height, width = image.shape[:2]
    target_height, target_width = target_size

    # Calculate padding amounts
    pad_y = max(target_height - height, 0)
    pad_x = max(target_width - width, 0)

    # Calculate padding for top, bottom, left, and right
    top_pad = pad_y // 2
    bottom_pad = pad_y - top_pad
    left_pad = pad_x // 2
    right_pad = pad_x - left_pad

    # Pad the image
    padded_image = np.pad(
        image,
        ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return padded_image

def zoom_out(rgb_image, depth_image, fx, fy, cx, cy, depth_scale):
    height, width = depth_image.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_image
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    z_new = z * depth_scale
    # z_new[z_new==0]= 1e-6

    u_new = (x * fx / z_new + cx).astype(np.float32)
    v_new = (y * fy / z_new + cy).astype(np.float32)

    depth_image_new = np.zeros_like(depth_image)
    depth_image_new[v.astype(int), u.astype(int)] = z_new

    map_x = u_new.astype(np.float32)
    map_y = v_new.astype(np.float32)
    rgb_image_new = cv2.remap(rgb_image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return rgb_image_new, depth_image_new

def get_traj(file_sheet, mass_csv):
    suc = pd.read_csv(file_sheet)
    suc = suc[~suc["Data ID 2"].isnull()]
    init = suc["Data ID 1"].to_list()
    mass = mass_csv

    successors = []
    traj = 0
    for i in range(len(init)):
        unexplored = 0
        for j in range(len(successors)):
            if init[i] not in successors[j]:
                unexplored += 1
        if unexplored == len(successors):
            cur_suc = init[i]
            successors.append([])
            while True:
                successors[traj].append(cur_suc)
                cur_suc = suc[suc["Data ID 1"] == successors[traj][-1]]["Data ID 2"]
                try:
                    cur_suc = cur_suc.iloc[0]
                except:
                    traj += 1
                    break
    successors = sorted(successors, key=lambda x: len(x))
    adjacent_pics = []
    for i in range(len(successors)):
        for jump in range(1,min(len(successors[i]),4)):
            for j in range(len(successors[i]) - jump):
                day_1 = successors[i][j]
                day_2 = successors[i][j + jump]
                if jump == 1:
                    delta = int(suc[suc["Data ID 1"] == day_1]["Delta"].iloc[0])
                else:
                    try:
                        mass_1 = int(mass[mass["Data ID"] == day_1]["Fresh Biomass"].iloc[0])
                        mass_2 = int(mass[mass["Data ID"] == day_2]["Fresh Biomass"].iloc[0])
                        delta = max(mass_2-mass_1,0)
                    except:
                        break
                new_set = [day_1, day_2, delta]
                date_1 = datetime.strptime(day_1[:10], "%Y-%m-%d")
                date_2 = datetime.strptime(day_2[:10], "%Y-%m-%d")
                new_set.append(int((date_2 - date_1).days))
                adjacent_pics.append(new_set)
    return adjacent_pics, successors

def test_data_128(coreDir, trainDir, trueDir, depthDir):
    trainPaths = [f for f in listdir(coreDir + trainDir)]
    trueData = pd.read_csv(coreDir + trueDir)
    trainPathswTrue = [
        (coreDir + trainDir + '/' + trainPaths[i], float(trueData[trueData['image'] == trainPaths[i]]['LFW'].iloc[0]))
        for i in range(len(trainPaths))]

    ## Set-up coupling with depth images and create 4-depth RGB-D images
    types = ['Flandria', 'Tiberius', 'Locarno']
    for i in tqdm.tqdm(range(len(trainPaths))):
        if trainPaths[i][0] == '1':
            trainPathswTrue[i] = (trainPathswTrue[i][0], trainPathswTrue[i][1],
                                  coreDir + depthDir + types[0] + '/' + 'Depth-' + trainPaths[i][:-4] + '.png')
        elif trainPaths[i][0] == '2':
            trainPathswTrue[i] = (trainPathswTrue[i][0], trainPathswTrue[i][1],
                                  coreDir + depthDir + types[1] + '/' + 'Depth-' + trainPaths[i][:-4] + '.png')
        elif trainPaths[i][0] == '3':
            trainPathswTrue[i] = (trainPathswTrue[i][0], trainPathswTrue[i][1],
                                  coreDir + depthDir + types[2] + '/' + 'Depth-' + trainPaths[i][:-4] + '.png')

    rgbdTrue = []
    size_ = 128
    for i in tqdm.tqdm(range(len(trainPaths))):
        try:
            rgb = np.asarray(Image.open(trainPathswTrue[i][0]).resize((size_, size_)))
            depth = np.asarray(Image.open(trainPathswTrue[i][2]))
            y, x = depth.shape
            startx = x // 2 - (size_ // 2)
            starty = y // 2 - (size_ // 2)
            depth = depth[starty:starty + size_, startx:startx + size_]
            depth = Image.fromarray(depth)
            depth = np.asarray(depth).reshape((size_, size_, -1))
            # print(rgb.shape, depth.shape)
            rgbd = np.concatenate([rgb, depth], axis=2)
            # rgbd = np.copy(rgb)
            rgbdTrue.append((rgbd, trainPathswTrue[i][1]))
        except Exception as error:
            print(error)
    return rgbdTrue

def test_data_third_auton(coreDir, trainDir, trueDir, depthDir):
    trainPaths = [f for f in listdir(coreDir + trainDir)]
    trueData = json.load(open(coreDir + trueDir))
    trainPathswTrue = []
    test_array = []

    for i in tqdm.tqdm(range(len(trainPaths))):
        try:
            num_ = trainPaths[i].replace('RGB_', '').replace('.png', '')
            if int(num_) <= 200:
                trainPathswTrue.append((coreDir + trainDir + '/' + trainPaths[i],
                                        trueData["Measurements"]["Image" + num_]["DryWeightShoot"],
                                        coreDir + depthDir + '/' + 'Depth_' + num_ + '.png'))
            test_array.append(trueData["Measurements"]["Image" + num_]["DryWeightShoot"])

        except:
            pass

    rgbdTrue = []
    for i in tqdm.tqdm(range(len(trainPathswTrue))):
        try:
            rgb = np.asarray(Image.open(trainPathswTrue[i][0]))[340:820, 695:1335]
            depth = np.asarray(Image.open(trainPathswTrue[i][2]))[340:820, 695:1335].reshape((480, 640, 1))

            target_size = (640, 640)
            rgb = padding_image(rgb, target_size)
            depth = padding_image(depth, target_size)
            # print(rgb.shape, depth.shape)
            rgbd = np.concatenate([rgb, depth], axis=2)
            rgbdTrue.append((rgbd, trainPathswTrue[i][1]))
        except Exception as error:
            print(error)
    return rgbdTrue

def test_data_my_data(coreDir, trainDir, trueDir, segmentation, artifZoom,banned,aug_factor, max_pics):
    trainPaths = sorted([f for f in listdir(coreDir + trainDir)])
    trueData = pd.read_csv(coreDir + trueDir)
    trainPathswTrue = [
        (coreDir + trainDir + '/' + trainPaths[i],
         float(trueData[trueData['Data ID'] == trainPaths[i].replace('.npy', '')]['Fresh Biomass'].iloc[0]),
         float(trueData[trueData['Data ID'] == trainPaths[i].replace('.npy', '')]['Distance (in mm)'].iloc[0]))
        for i in range(len(trainPaths))]

    if not artifZoom:
        trainPaths_filt = []
        for i in range(len(trainPathswTrue)):
            if trainPathswTrue[i][2] == 176 and trainPaths[i] not in banned:
                # print(trainPaths[i])
                trainPaths_filt.append(trainPathswTrue[i])
        trainPathswTrue = trainPaths_filt[:]
    # print([trainPathswTrue[i][2] for i in range(len(trainPathswTrue))])
    random.shuffle(trainPathswTrue)

    rgbdTrue = []
    for i in tqdm.tqdm(range(max_pics//aug_factor)):
        try:
            rgbd = np.load(trainPathswTrue[i][0])
            target_size = (640, 640)
            depth_image = rgbd[:, :, 3].astype(np.float32)
            # print(np.sum(depth_image == 0)/(640*640))
            mask = (depth_image == 0).astype(np.uint8)
            smoothed_depth = cv2.inpaint(depth_image.astype(np.float32), mask, inpaintRadius=10,
                                         flags=cv2.INPAINT_TELEA)
            rgbd = np.concatenate((rgbd[:, :, :3], smoothed_depth.reshape((480, 640, 1))), axis=2)
            rgbd = padding_image(rgbd, target_size)
            if segmentation:
                rgb = rgbd[:, :, :3].astype(np.uint8)
                rgb_ = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                hsv = cv2.cvtColor(rgb_, cv2.COLOR_RGB2HSV)
                lower_green = np.array([35, 35, 35])
                upper_green = np.array([85, 255, 255])

                mask = cv2.inRange(hsv, lower_green, upper_green)
                kernel = np.ones((5, 5), np.uint8)
                mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                kernel_dilation = np.ones((5, 5), np.uint8)
                mask_opened = cv2.dilate(mask_opened, kernel_dilation, iterations=1)

                result = cv2.bitwise_and(rgb_, rgb_, mask=mask_opened)
                filter_depth = 65535
                depth = rgbd[:, :, 3]
                depth[mask_opened == 0] = filter_depth

                rgbd = np.concatenate((cv2.cvtColor(result, cv2.COLOR_RGB2BGR), depth.reshape((640, 640, 1))), axis=2)
            if artifZoom:
                rgb_image = rgbd[:, :, :3].astype(np.uint8)
                depth_image = rgbd[:, :, 3].astype(np.float32)

                fx, fy = 424.035, 424.035
                cx, cy = 421.168, 238.846

                depth_scale = trainPathswTrue[i][2] / 198

                zoomed_rgb, zoomed_depth = zoom_out(rgb_image, depth_image, fx, fy, cx, cy, depth_scale)
                rgbd = np.concatenate(
                    (cv2.cvtColor(zoomed_rgb, cv2.COLOR_RGB2BGR), zoomed_depth.reshape((640, 640, 1))), axis=2)

            rgbdTrue.append((rgbd, trainPathswTrue[i][1]))
        except Exception as error:
            print(error)
    return rgbdTrue

def test_data_my_data_delta(coreDir, trainDir, trueDir, suc_sheet, segmentation, artifZoom, banned,jump_amt, aug_factor, max_pics):
    trueData = pd.read_csv(coreDir + trueDir)
    adj_,_ = get_traj(suc_sheet, trueData)
    adj_ = isolate_test_data(adj_, banned, 'delta')

    adj_filt = []
    for i in range(len(adj_)):
        try:
            h_1 = float(trueData[trueData['Data ID'] == adj_[i][0].replace('.npy', '')]['Distance (in mm)'].iloc[0])
            h_2 = float(trueData[trueData['Data ID'] == adj_[i][1].replace('.npy', '')]['Distance (in mm)'].iloc[0])
            if not artifZoom and (h_1 == 176 and h_2 == 176):
                adj_filt.append(adj_[i])
            elif artifZoom:
                adj_filt.append([adj_[i][0], adj_[i][1], adj_[i][2], h_1, h_2])
        except:
            pass
    adj_ = adj_filt[:]
    if artifZoom:
        trainPathswTrue = [
            (coreDir + trainDir + adj[0] + '.npy', coreDir + trainDir + adj[1] + '.npy', adj[2], adj[3], adj[4]) for adj
            in adj_]
    else:
        trainPathswTrue = [(coreDir + trainDir + adj[0] + '.npy', coreDir + trainDir + adj[1] + '.npy', adj[2]) for adj
                           in adj_]
    random.shuffle(trainPathswTrue)
    rgbdTrue = []
    for i in tqdm.tqdm(range(max_pics//aug_factor)):
        rgbds = []
        try:
            for j in range(2):
                rgbd = np.load(trainPathswTrue[i][j])
                target_size = (640, 640)
                depth_image = rgbd[:, :, 3].astype(np.float32)
                # print(np.sum(depth_image == 0)/(640*640))
                mask = (depth_image == 0).astype(np.uint8)
                smoothed_depth = cv2.inpaint(depth_image.astype(np.float32), mask, inpaintRadius=10,
                                             flags=cv2.INPAINT_TELEA)
                rgbd = np.concatenate((rgbd[:, :, :3], smoothed_depth.reshape((480, 640, 1))), axis=2)
                rgbd = padding_image(rgbd, target_size)
                if segmentation:
                    rgb = rgbd[:, :, :3].astype(np.uint8)
                    rgb_ = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    hsv = cv2.cvtColor(rgb_, cv2.COLOR_RGB2HSV)
                    lower_green = np.array([35, 35, 35])
                    upper_green = np.array([85, 255, 255])

                    mask = cv2.inRange(hsv, lower_green, upper_green)
                    kernel = np.ones((5, 5), np.uint8)
                    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    kernel_dilation = np.ones((5, 5), np.uint8)
                    mask_opened = cv2.dilate(mask_opened, kernel_dilation, iterations=1)

                    result = cv2.bitwise_and(rgb_, rgb_, mask=mask_opened)
                    filter_depth = 65535
                    depth = rgbd[:, :, 3]
                    depth[mask_opened == 0] = filter_depth

                    rgbd = np.concatenate((cv2.cvtColor(result, cv2.COLOR_RGB2BGR), depth.reshape((640, 640, 1))),
                                          axis=2)
                if artifZoom:
                    rgb_image = rgbd[:, :, :3].astype(np.uint8)
                    depth_image = rgbd[:, :, 3].astype(np.float32)

                    fx, fy = 424.035, 424.035
                    cx, cy = 421.168, 238.846

                    depth_scale = trainPathswTrue[i][3 + j] / 198

                    zoomed_rgb, zoomed_depth = zoom_out(rgb_image, depth_image, fx, fy, cx, cy, depth_scale)
                    rgbd = np.concatenate(
                        (cv2.cvtColor(zoomed_rgb, cv2.COLOR_RGB2BGR), zoomed_depth.reshape((640, 640, 1))), axis=2)
                rgbds.append(rgbd)
            rgbd_full = np.concatenate((rgbds[0], rgbds[1]), axis=2)
            rgbdTrue.append((rgbd_full, trainPathswTrue[i][2]))
        except Exception as error:
            print(error)
    return rgbdTrue

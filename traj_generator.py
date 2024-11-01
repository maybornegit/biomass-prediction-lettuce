import pandas as pd
import torch,tqdm,os,cv2
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from biomass_cnn_models import CNN_640, CNN_Delta, ResNet640
import prediction_utils as p_util
from data_utils import get_traj

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

def delta_stack(data, banned):
    coreDir = os.path.expanduser("~/Downloads/CustomI2GROW_Dataset")
    trainDir = "/RGBDImages/"
    trueDir = "/Biomass_Info_Ground_Truth.csv"
    trueData = pd.read_csv(coreDir + trueDir)


    adj_filt = []
    for i in range(len(data)):
        try:
            h_1 = float(trueData[trueData['Data ID'] == data[i][0].replace('.npy', '')]['Distance (in mm)'].iloc[0])
            h_2 = float(trueData[trueData['Data ID'] == data[i][1].replace('.npy', '')]['Distance (in mm)'].iloc[0])
            if (data[i][0] not in banned and data[i][1] not in banned) and (h_1 == 176 and h_2 == 176):
                # print(trainPaths[i])
                adj_filt.append(data[i])
        except:
            pass
    data = adj_filt[:]

    trainPathswTrue = [(coreDir+trainDir+d[0]+'.npy',coreDir+trainDir+d[1]+'.npy', d[2]) for d in data]

    rgbdTrue = []
    for i in tqdm.tqdm(range(len(trainPathswTrue))):
        rgbds = []
        try:

            for j in range(2):
                rgbd = np.load(trainPathswTrue[i][j])
                target_size = (640, 640)
                depth_image = rgbd[:, :, 3].astype(np.float32)
                mask = (depth_image == 0).astype(np.uint8)
                smoothed_depth = cv2.inpaint(depth_image.astype(np.float32), mask, inpaintRadius=10,
                                             flags=cv2.INPAINT_TELEA)
                rgbd = np.concatenate((rgbd[:, :, :3], smoothed_depth.reshape((480, 640, 1))), axis=2)
                rgbd = padding_image(rgbd, target_size)
                rgbds.append(rgbd)
            rgbd_full = np.concatenate((rgbds[0],rgbds[1]), axis=2)
            rgbdTrue.append((rgbd_full,trainPathswTrue[i][2],trainPathswTrue[i][0][57:-4],trainPathswTrue[i][1][57:-4]))
        except Exception as error:
            print(error)
    return rgbdTrue

def filter_data(banned,segmentation=True):
    coreDir = os.path.expanduser("~/Downloads/CustomI2GROW_Dataset")
    trainDir = "/RGBDImages/"
    trueDir = "/Biomass_Info_Ground_Truth.csv"
    trueData = pd.read_csv(coreDir + trueDir)
    data = sorted([f for f in listdir(coreDir + trainDir)])

    pic_filt = []
    for i in range(len(data)):
        try:
            h_1 = float(trueData[trueData['Data ID'] == data[i].replace('.npy', '')]['Distance (in mm)'].iloc[0])
            if (data[i] not in banned) and (h_1 == 176):
                # print(trainPaths[i])
                pic_filt.append(data[i])
        except:
            pass
    data = pic_filt[:]

    trainPathswTrue = [(coreDir + trainDir + d, float(trueData[trueData['Data ID'] == d.replace('.npy', '')]['Fresh Biomass'].iloc[0])) for d in data]

    rgbdTrue = []
    for i in tqdm.tqdm(range(len(trainPathswTrue))):
        try:
            rgbd = np.load(trainPathswTrue[i][0])
            target_size = (640, 640)
            depth_image = rgbd[:, :, 3].astype(np.float32)
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
            rgbdTrue.append((rgbd, trainPathswTrue[i][1], trainPathswTrue[i][0][57:-4]))
        except Exception as error:
            print(error)
    return rgbdTrue

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle')

### Relevant Paths
suc_sheet = '/home/frc-ag-2/Downloads/CustomI2GROW_Dataset/successor_sheet.csv'
delta_path = '/home/frc-ag-2/Downloads/old_cnn_results/20241010-deltawtest-jump3.pt'
cnn_path = '/home/frc-ag-2/Downloads/old_cnn_results/20241017-standwtest-resnet-mape-1.pt'
lstm_paths = ['/home/frc-ag-2/Downloads/old_cnn_results/lstm_20241010-deltawtest-jump3-len'+str(j)+'-delta.h5' for j in range(4,9)]

coreDir = os.path.expanduser("~/Downloads/CustomI2GROW_Dataset")
trueDir = "/Biomass_Info_Ground_Truth.csv"
trueData = pd.read_csv(coreDir + trueDir)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the CPU")

### CNN Models
net_d = CNN_Delta().to(device)
net_d.load_state_dict(torch.load(delta_path, weights_only=True))
net_d.eval()

net_c = ResNet640().to(device)
rmse = .2529*19.833
net_c.load_state_dict(torch.load(cnn_path, weights_only=True))
net_c.eval()

### Relevant Trajectory Loading
data, trajs = get_traj(suc_sheet, trueData)
test_data_names = ['2024-09-10-0', '2024-09-11-0', '2024-09-12-0', '2024-09-13-0','2024-09-16-0', '2024-09-17-0', '2024-09-18-0', '2024-09-19-0', '2024-09-20-0',
              '2024-09-10-3','2024-09-11-3', '2024-09-12-3', '2024-09-13-3', '2024-09-16-3', '2024-09-17-3', '2024-09-18-3','2024-09-19-7', '2024-09-20-7',
              '2024-09-10-7', '2024-09-11-7', '2024-09-12-7', '2024-09-13-7','2024-09-16-7', '2024-09-17-7', '2024-09-18-7', '2024-09-19-11', '2024-09-20-11', '2023-09-24-4', '2024-09-25-3', '2024-09-26-3', '2024-09-27-3']
### Filtering Trajectories
trajs_test = []
for i in range(len(trajs)):
    if trajs[i][0] in test_data_names:
        trajs_test.append(trajs[i])
trajs = trajs_test[:]

### Filtering Raw Data
test_data = []
for i in range(len(data)):
    if data[i][0] in test_data_names or data[i][1] in test_data_names:
        test_data.append(data[i])
data = test_data[:]

### Filtering Bad Data
banned = ['2024-09-17-14']
net_c_data = filter_data(banned) ## Only single-height
net_d_data = delta_stack(data,banned)

### Raw Predictions
oupts_delta, oupts_cnn = p_util.pred_all(net_d_data,net_c_data, net_c, net_d)

### Predictions Mapped to Trajectories
traj_pred = []
for i in range(len(trajs)):
    traj_array_d = list(p_util.pred_traj_delta(trajs[i], trueData, net_d_data, oupts_delta))
    traj_array_c = list(p_util.pred_traj_cnn(trajs[i], net_c_data, oupts_cnn))
    traj_array_lstm = list(p_util.pred_traj_lstm(traj_array_d[:], lstm_paths, avg=False))
    traj_array_g = list(p_util.ground_truth_traj(trajs[i],trueData))
    print(np.array(traj_array_c[0]), np.array(traj_array_c[1]))
    traj_pred_bayes = list(p_util.bayesian_regression(np.array(traj_array_c[0]), np.array(traj_array_c[1]), rmse))
    traj_pred.append([i, traj_array_c,traj_array_d, traj_array_g, traj_array_lstm,traj_pred_bayes])

# print([traj[1] for traj in traj_pred]) # CNN Prediction
# print([traj[2] for traj in traj_pred]) # Delta Prediction
# print([traj[3] for traj in traj_pred]) # LSTM Prediction
# print([traj[4] for traj in traj_pred]) # Ground Truth
# print([len(traj[4][0]) for traj in traj_pred])

### Plot Trajectories
for i in range(len(traj_pred)):
    map_traj = traj_pred[i]
    plt.figure(i)
    plt.plot(np.array(map_traj[1][0]),np.array(map_traj[1][1]), label="CNN Prediction")
    # plt.plot(np.array(map_traj[2][0]),np.array(map_traj[2][1]), label="Delta Prediction")
    plt.plot(np.array(map_traj[3][0]), np.array(map_traj[3][1]), label="Ground Truth")
    # plt.plot(np.array(map_traj[4][0]), np.array(map_traj[4][1]), label="Delta-LSTM Prediction")
    plt.plot(np.array(map_traj[5][0]),np.array(map_traj[5][1]), label="Base-Curve Fit Prediction")
    plt.xlabel("Days")
    plt.ylabel("Biomass")
    plt.title("Trajectories")
    plt.legend()

traj_len = min([len(traj[4][0]) for traj in traj_pred])
errors = []
for i in range(len(trajs)):
    errors.append([list(np.array(traj_pred[i][j][1])-np.array(traj_pred[i][3][1])) for j in [1,2,4,5]])

stat_info = [[[],[],[],[]] for i in range(5)]
for i in range(traj_len):
    stat_info[0][0].append(np.mean([traj[1][1][i] for traj in traj_pred]))
    stat_info[1][0].append(np.mean([traj[2][1][i] for traj in traj_pred]))
    stat_info[2][0].append(np.mean([traj[3][1][i] for traj in traj_pred]))
    stat_info[3][0].append(np.mean([traj[4][1][i] for traj in traj_pred]))
    stat_info[4][0].append(np.mean([traj[5][1][i] for traj in traj_pred]))

    stat_info[0][1].append(np.std([traj[1][1][i] for traj in traj_pred]))
    stat_info[1][1].append(np.std([traj[2][1][i] for traj in traj_pred]))
    stat_info[2][1].append(np.std([traj[3][1][i] for traj in traj_pred]))
    stat_info[3][1].append(np.std([traj[4][1][i] for traj in traj_pred]))
    stat_info[4][1].append(np.std([traj[5][1][i] for traj in traj_pred]))

    stat_info[0][2].append(np.mean([error[0][i] for error in errors]))
    stat_info[1][2].append(np.mean([error[1][i] for error in errors]))
    stat_info[2][2].append(np.mean([0 for error in errors]))
    stat_info[3][2].append(np.mean([error[2][i] for error in errors]))
    stat_info[4][2].append(np.mean([error[3][i] for error in errors]))

    stat_info[0][3].append(np.std([error[0][i] for error in errors]))
    stat_info[1][3].append(np.std([error[1][i] for error in errors]))
    stat_info[2][3].append(np.std([0 for error in errors]))
    stat_info[3][3].append(np.std([error[2][i] for error in errors]))
    stat_info[4][3].append(np.std([error[3][i] for error in errors]))

plt.figure(len(traj_pred))
names = ["Base CNN", "Delta CNN", "Ground Truth","Delta+LSTM CNN","Base CNN + Curve Fit "]
for i in [2,0]:
    alpha = 0.3 if i != 2 else 0.8
    mean = stat_info[i][0]
    ci = 1.96 * np.array(stat_info[i][1]) / np.sqrt(traj_len)
    plt.plot(np.array(traj_pred[0][1][0]),np.array(mean), label=names[i])
    plt.fill_between(np.array(traj_pred[0][1][0]), np.array(mean)-ci,np.array(mean)+ci, alpha=alpha)
plt.xlabel("Days")
plt.ylabel("Biomass")
plt.title("Mean Trajectories")
plt.legend()

plt.figure(len(traj_pred)+1)
for i in [2,4]:
    alpha = 0.3 if i != 2 else 0.8
    mean = stat_info[i][0]
    ci = 1.96 * np.array(stat_info[i][1]) / np.sqrt(traj_len)
    plt.plot(np.array(traj_pred[0][1][0]),np.array(mean), label=names[i])
    plt.fill_between(np.array(traj_pred[0][1][0]), np.array(mean)-ci,np.array(mean)+ci, alpha=alpha)
plt.xlabel("Days")
plt.ylabel("Biomass")
plt.title("Mean Trajectories")
plt.legend()


plt.figure(len(traj_pred)+2)
# for i in range(len(stat_info)):
for i in [0,4,2]:
    error = stat_info[i][2]
    if i != 2:
        ci = 1.96 * np.array(stat_info[i][3]) / np.sqrt(traj_len)
        plt.plot(np.array(traj_pred[0][1][0]),np.array(error), label=names[i])
        plt.fill_between(np.array(traj_pred[0][1][0]), np.array(error) - ci, np.array(error) + ci, alpha=0.3)
    else:
        plt.plot(np.array(traj_pred[0][1][0]), np.array(error), label=names[i])
plt.xlabel("Days")
plt.ylabel("Biomass")
plt.title("Error")
plt.legend()
plt.show()
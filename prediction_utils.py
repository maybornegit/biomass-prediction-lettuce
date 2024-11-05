import torch, tqdm
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit
from keras.models import load_model

def pred_all(net_d_data,net_c_data, net_c, net_d):
    oupts_delta = []
    for i in tqdm.tqdm(range(len(net_d_data))):
        X = (torch.from_numpy(np.transpose(net_d_data[i][0], (2, 0, 1))).float())
        # X = torch.from_numpy(net_d_data[i][0]).float()
        X = torch.reshape(X, (1, 8, 640, 640))
        if torch.cuda.is_available():
            X = X.to('cuda')
        oupt = net_d(X).detach().reshape((-1,))  # shape [10,1]
        oupts_delta.append((float(oupt), net_d_data[i][1]))

    oupts_cnn = []
    for i in tqdm.tqdm(range(len(net_c_data))):
        X = (torch.from_numpy(np.transpose(net_c_data[i][0], (2, 0, 1))).float())
        # X = torch.from_numpy(net_c_data[i][0]).float()
        X = torch.reshape(X, (1, 4, 640, 640))
        if torch.cuda.is_available():
            X = X.to('cuda')
        oupt = net_c(X).detach().reshape((-1,))  # shape [10,1]
        oupts_cnn.append((float(oupt), net_c_data[i][1]))

    return oupts_delta, oupts_cnn

def pred_traj_delta(seq, trueData, net_d_data, oupts_delta):
    first_pic = [d[2] for d in net_d_data]
    sec_pic = [d[3] for d in net_d_data]
    traj_prediction = []
    traj_prediction.append((float(trueData[trueData['Data ID'] == seq[0].replace('.npy', '')]['Fresh Biomass'].iloc[0]), seq[0]))
    for j in range(len(seq) - 1):
        if seq[j] in first_pic and seq[j + 1] in sec_pic:
            idx = sec_pic.index(seq[j + 1])
            traj_prediction.append((traj_prediction[-1][0] + oupts_delta[idx][0], seq[j + 1]))
        else:
            break
        if j == len(seq) - 2:
            break
    times = []
    traj = []
    for j in range(len(seq)):
        if j == 0:
            times.append(2)
        else:
            diff = (datetime.strptime(seq[j][:10], "%Y-%m-%d") - datetime.strptime(seq[j-1][:10],
                                                                                           "%Y-%m-%d")).days
            times.append(times[-1] + int(diff))
        traj.append(traj_prediction[j][0])
    return times, traj

def pred_traj_cnn(seq, net_c_data, oupts_cnn):
    pics = [d[2] for d in net_c_data]
    traj_prediction = []
    for j in range(len(seq)):
        if seq[j] in pics:
            idx = pics.index(seq[j])
            traj_prediction.append((oupts_cnn[idx][0], seq[j]))
        else:
            break
        if j == len(seq) - 1:
            break

    times = []
    traj = []
    for j in range(len(seq)):
        if j == 0:
            times.append(2)
        else:
            diff = (datetime.strptime(seq[j][:10], "%Y-%m-%d") - datetime.strptime(seq[j-1][:10],
                                                                                           "%Y-%m-%d")).days
            times.append(times[-1] + int(diff))
        traj.append(traj_prediction[j][0])
    return times, traj

def ground_truth_traj(seq,trueData):
    traj_array_g = [[], []]
    for j in range(len(seq)):
        traj_array_g[1].append(
            float(trueData[trueData['Data ID'] == seq[j].replace('.npy', '')]['Fresh Biomass'].iloc[0]))
        if j == 0:
            traj_array_g[0].append(10)
        else:
            diff = (datetime.strptime(seq[j][:10], "%Y-%m-%d") - datetime.strptime(seq[j - 1][:10],
                                                                                        "%Y-%m-%d")).days
            traj_array_g[0].append(traj_array_g[0][-1] + int(diff))
        if (traj_array_g[0][-1] - 10) >= 14:
            break
    return traj_array_g[0], traj_array_g[1]

def pred_traj_lstm(pred_traj, model_paths, avg=True):
    new_traj = pred_traj[1][:]
    for j in range(4,8):
        model_pred = load_model(model_paths[j-4])
        predictions = model_pred.predict(np.array(new_traj[:j]).reshape((1,j,1)))
        if avg:
            new_traj[j+1] = (new_traj[j+1] + predictions[0][0])/2
        else:
            new_traj[j + 1] = predictions[0][0]

    return pred_traj[0],new_traj

def bayesian_regression(t, pred, rmse):
    def exponential_model(t, a, b):
        return a * np.exp(b * t)

    print(rmse)
    params, covariance = curve_fit(exponential_model, t, pred, p0=(4, 0.2), sigma=np.full_like(pred, rmse))

    # Extract fitted parameters
    a_fit, b_fit = params
    print(f"Fitted parameters: a = {a_fit}, b = {b_fit}")

    # Generate fitted values
    y_fit = exponential_model(t, a_fit, b_fit)
    return list(t), list(y_fit)
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import sys

from localization.utils import *
from data_reader import Girona
from localization.inekf import InEKF

from localization.frame_utils import *


def main_girona_idp(data, X0):
    filter_data = InEKF(X0)

    _X_pred = []
    _time_pred = []

    filter_data.Prediction(data.y_imu[:,0], 0.1)
    i_ts_imu = 1
    i_ts_dvl = 0
    i_ts_depth = 0
    
    ts_curr = min(data.ts_imu[0, 0], data.ts_dvl[0, 0], data.ts_depth[0, 0])
    
    while (i_ts_imu < data.len_imu and i_ts_dvl < data.len_dvl \
            and i_ts_depth < data.len_depth): #
        if ((data.ts_imu[0, i_ts_imu] < data.ts_dvl[0, i_ts_dvl]) \
            and (data.ts_imu[0, i_ts_imu] < data.ts_depth[0, i_ts_depth])):
            # Update w/ IMU
            # imu: omega, acc
            dt = data.ts_imu[0,i_ts_imu] - data.ts_imu[0,i_ts_imu-1]
            u_input = 0.5*(data.y_imu[:, i_ts_imu-1] + data.y_imu[:, i_ts_imu])
            filter_data.Prediction(u_input, dt)
            ts_curr = data.ts_imu[0,i_ts_imu]
            i_ts_imu += 1
        
        if ((data.ts_dvl[0, i_ts_dvl] <= data.ts_imu[0, i_ts_imu]) \
            and (i_ts_dvl < data.len_dvl) ):
            # Update DVL
            y_dvl =  np.matmul(data.uuv.R_dvl_imu, data.y_dvl[:,i_ts_dvl]) + np.matmul(data.uuv.t_dvl_imu_skew, (data.y_imu[:3,i_ts_imu]))
            filter_data.CorrectionDvl(y_dvl)
            ts_curr = data.ts_dvl[0, i_ts_dvl]
            i_ts_dvl += 1
            
        if ((data.ts_depth[0, i_ts_depth] <= data.ts_imu[0, i_ts_imu])):
            # Update depth
            y_depth = -(data.y_depth[0, i_ts_depth] + data.uuv.t_depth_imu_z)           
            filter_data.CorrectionDepth(y_depth) 
            ts_curr = data.ts_depth[0, i_ts_depth]
            i_ts_depth += 1
            
        # Save prediction
        _X_pred.append(filter_data.X)
        _time_pred.append(ts_curr)
        
        if (i_ts_imu%2000 == 0):
            print("IMU timestamp index: " + str(i_ts_imu))

    _X_pred = np.array(_X_pred)
    _time_pred = np.array(_time_pred)
    
    return _X_pred, _time_pred


def main_girona_idpc(data, X0):
    yolo = KeyFramesYolo()
    mono_frame = MonoFrame()
    filter_data = InEKF(X0)

    _X_pred = []
    _time_pred = []

    filter_data.Prediction(data.y_imu[:,0], 0.1)
    i_ts_imu = 1
    i_ts_dvl = 0
    i_ts_depth = 0
    i_ts_cam = 0
    
    ts_curr = min(data.ts_imu[0, 0], data.ts_dvl[0, 0], data.ts_depth[0, 0])
    
    while (i_ts_imu < data.len_imu and i_ts_dvl < data.len_dvl \
            and i_ts_depth < data.len_depth and i_ts_cam < data.len_cam ): #
        if ((data.ts_imu[0, i_ts_imu] < data.ts_dvl[0, i_ts_dvl]) \
            and (data.ts_imu[0, i_ts_imu] < data.ts_depth[0, i_ts_depth]) \
            and (data.ts_imu[0, i_ts_imu] < data.ts_cam[0, i_ts_cam]) ):
            # Update w/ IMU
            # imu: omega, acc
            dt = data.ts_imu[0,i_ts_imu] - data.ts_imu[0,i_ts_imu-1]
            u_input = 0.5*(data.y_imu[:, i_ts_imu-1] + data.y_imu[:, i_ts_imu])
            filter_data.Prediction(u_input, dt)
            ts_curr = data.ts_imu[0,i_ts_imu]
            i_ts_imu += 1
        
        if ((data.ts_dvl[0, i_ts_dvl] <= data.ts_imu[0, i_ts_imu]) \
            and (i_ts_dvl < data.len_dvl) ):
            # Update DVL
            y_dvl =  np.matmul(data.uuv.R_dvl_imu, data.y_dvl[:,i_ts_dvl]) + np.matmul(data.uuv.t_dvl_imu_skew, (data.y_imu[:3,i_ts_imu]))
            filter_data.CorrectionDvl(y_dvl)
            ts_curr = data.ts_dvl[0, i_ts_dvl]
            i_ts_dvl += 1
            
        if ((data.ts_depth[0, i_ts_depth] <= data.ts_imu[0, i_ts_imu])):
            # Update depth
            y_depth = -(data.y_depth[0, i_ts_depth] + data.uuv.t_depth_imu_z)           
            filter_data.CorrectionDepth(y_depth) 
            ts_curr = data.ts_depth[0, i_ts_depth]
            i_ts_depth += 1
            
        if ((data.ts_cam[0, i_ts_cam] <= data.ts_imu[0, i_ts_imu]) \
            and (i_ts_cam < data.len_cam) ):
            # Update Camera
            
            if (i_ts_cam == 0):
                img_curr = GetImage(data, i_ts_cam)
                i_ts_curr_cam_at_imu = i_ts_imu
                R_curr_cam = filter_data.X[:3, :3]
            else:
                img_prev = img_curr
                i_ts_prev_cam_at_imu = i_ts_curr_cam_at_imu
                i_ts_curr_cam_at_imu = i_ts_imu
                R_prev_cam = R_curr_cam
                R_curr_cam = filter_data.X[:3, :3]
                img_curr = GetImage(data, i_ts_cam)
                n_matches, R, _, _, _ = PoseEstimate2D2D(img_prev, img_curr, mono_frame.orb, mono_frame.K)
                if (n_matches > 420):
                    filter_data.CorrectionCam(R_prev_cam, R)
 
            ts_curr = data.ts_cam[0, i_ts_cam]
            i_ts_cam += 1
            
        # Save prediction
        _X_pred.append(filter_data.X)
        _time_pred.append(ts_curr)
        
        if (i_ts_imu%2000 == 0):
            print("IMU timestamp index: " + str(i_ts_imu))

    _X_pred = np.array(_X_pred)
    _time_pred = np.array(_time_pred)
    
    return _X_pred, _time_pred


def SingleRun():
    data = Girona()
    data.GetData()
    
    X0 = np.eye(5)
    X0[:3,:3] = data.R_init
    X0[:3,4:] = data.p_init

    X_pred, time_pred = main_girona_idp(data, X0)
    np.savez('./results/girona_idp_one.npz', X_pred = X_pred, time_pred = time_pred)
    print('IMU-DVL-Pressure Result Saved.')

    X_pred, time_pred = main_girona_idpc(data, X0)
    np.savez('./results/girona_idpc_one.npz', X_pred = X_pred, time_pred = time_pred)
    print('IMU-DVL-Pressure-Camera Result Saved.')    

    
def MultiRun():
    data = Girona()
    data.GetData()
    
    all_X = []
    all_t = []
    
    for i in range(5):
        X0 = np.eye(5)
        q = data.quat_init + np.random.normal(0, 0.2, size=(1, 4))
        R = GetRotMatFromQuat(q)
        X0[:3,:3] = R
        X0[:3,4:] = data.p_init + np.random.normal(0, 2, size=(3, 1))
        
        X_pred, time_pred = main_girona(data, X0) 
        all_X.append(X_pred)
        all_t.append(time_pred)
        print(str(i) + " done")
    
    all_X = np.array(all_X)
    all_t = np.array(all_t)
    
    np.savez('./results/girona_idpc_multi_run.npz', all_X = all_X, all_t = all_t)
    print("data saved.")    


if __name__ == "__main__":
    SingleRun()
    #MultiRun()


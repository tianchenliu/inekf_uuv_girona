#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from localization.utils import *
from localization.vehicle import Uuv_Girona

import os

class Girona:
    def __init__(self):
        data_folder = os.getcwd() + "/../data/girona/"
        self.imu_file = data_folder + "full_dataset/imu_adis.txt"
        self.depth_file = data_folder + "full_dataset/depth_sensor.txt"
        self.dvl_file = data_folder + "full_dataset/dvl_linkquest.txt"
        self.odom_file = data_folder + "full_dataset/odometry.txt"
        self.tf_file = data_folder + "full_dataset/tf.txt"
        self.ts_img_file = data_folder + "frames_timestamps.txt" 
        self.img_folder = data_folder + "frames/"        
        self.uuv = Uuv_Girona()

    def GetOdomData(self):
        data = np.array(pd.read_csv(self.odom_file))
        self.len_odom = len(data)
        self.ts_odom = np.zeros((1, self.len_odom))
        self.y_odom_p = np.zeros((3, self.len_odom))
        self.y_odom_quat = np.zeros((4, self.len_odom))     
        self.y_odom = np.zeros((7, self.len_odom))
        
        for i in range(self.len_odom):
            self.ts_odom[:, i] = data[i, 0]
            self.y_odom_p[:, i] = data[i, 3:6]
            self.y_odom_quat[:, i] = data[i, 6:10]
            self.y_odom[:, i] = [data[i,3], data[i,4], data[i,5], data[i,6], data[i,7], data[i,8], data[i,9]] 
        self.ts_odom = self.ts_odom * 1e-9

    def GetDepthData(self):
        data = np.array(pd.read_csv(self.depth_file))
        self.len_depth = len(data)
        self.ts_depth = np.zeros((1, self.len_depth))
        self.y_depth = np.zeros((1, self.len_depth))
        for i in range(self.len_depth):
            self.ts_depth[:, i] = data[i, 0]
            self.y_depth[:, i] = data[i, 3]
        
        self.ts_depth = self.ts_depth * 1e-9

    def GetDVLData(self):
        data = np.array(pd.read_csv(self.dvl_file))
        self.len_dvl = len(data)
        self.ts_dvl = np.zeros((1, self.len_dvl))
        self.y_dvl = np.zeros((3, self.len_dvl))
        for i in range(self.len_dvl):
            self.ts_dvl[:, i] = data[i, 0]
            self.y_dvl[:, i] = data[i, 23:26] 

        self.ts_dvl = self.ts_dvl * 1e-9
    
    def GetImuData(self):
        data = np.array(pd.read_csv(self.imu_file))
        self.len_imu = len(data)
        self.ts_imu = np.zeros((1, self.len_imu))
        self.y_imu_euler = np.zeros((3, self.len_imu))
        self.y_imu_quat = np.zeros((4, self.len_imu))
        self.y_omega = np.zeros((3, self.len_imu))
        self.y_acc = np.zeros((3, self.len_imu))
        self.y_imu = np.zeros((6, self.len_imu))
        for i in range(self.len_imu):
            self.ts_imu[:, i] = data[i, 0]
            self.y_imu_euler[:, i] = data[i, 3:6]
            self.y_imu_quat[:, i] = data[i, 6:10]
        
            omega1 = np.array(data[i, 17:20]) - np.array(data[i, 20:23])
            
            self.y_omega[:, i] = omega1 
            self.y_acc[:, i] = data[i, 14:17] 
            self.y_imu[:, i] = [self.y_omega[0, i], self.y_omega[1, i], self.y_omega[2, i] , \
                                self.y_acc[0, i], self.y_acc[1, i], self.y_acc[2, i]] # omega, acc 
        self.quat_init = data[0, 6:10]

        self.ts_imu = self.ts_imu * 1e-9
            

    def GetInitPose(self):
        self.R_init = GetRotMatFromQuat(self.quat_init)
        self.p_init = np.zeros((3, 1))
        self.p_init[2, 0] = -self.y_depth[0, 0]

    def GetCamData(self):
        data = np.array(pd.read_csv(self.ts_img_file, sep='\t'))
        self.len_cam = len(data)
        self.ts_cam = np.zeros((1, self.len_cam))
        self.files_cam = []
        for i in range(self.len_cam):
            self.files_cam.append(data[i, 0])
            self.ts_cam[:, i] = data[i, 1]

    def GetData(self):
        self.GetOdomData()
        self.GetDepthData()
        self.GetDVLData()
        self.GetImuData()
        self.GetInitPose()
        self.GetCamData()


if __name__ == "__main__":
    data = Girona()
    data.GetData()

    #print(data.R_init)
    #print(data.p_init)

        
        







import numpy as np
from scipy.linalg import expm, block_diag
import cv2

from localization.utils import *
from localization.vehicle import Uuv_Girona

from numpy.linalg import pinv, inv

class InEKF:

    def __init__(self, X0):
        self.X = X0 
        self.P = 0.1 * np.eye(9)  # state covariance
        
        self.Q_omega = 0.1*np.eye(3)
        self.Q_a = 0.1*np.eye(3)
        self.Q = block_diag(self.Q_omega, self.Q_a, np.zeros((3,3)))
        
        self.g = np.array([[0, 0, -9.81]]).T

        self.A = np.zeros((9,9))
        self.A[3:6,0:3] = SkewMatrix(self.g)
        self.A[6:,3:6] = np.eye(3)
        
        self.uuv = Uuv_Girona()


    def Prediction(self, u, dt):
        omega = u[:3]
        omega = np.array([omega]).T # 3x1
        acc = u[3:6]
        acc = np.array([acc]).T # 3x1
        
        Phi = expm(self.A*dt)   
        Qd = (Phi @ self.Q @ Phi.T) *dt  
        AdjX = AdjointSE2_3(self.X)
        self.P = Phi @ self.P @ Phi.T + AdjX @ Qd @ AdjX.T

        R, v, p, m = ChiToState(self.X)
        phi = omega*dt
        R_pred = np.matmul(R, ExpSO3(phi)) # 3x3
        v_pred = v + (np.matmul(R, acc) + self.g)*dt # 3x1
        p_pred = p + v*dt + 0.5*(np.matmul(R, acc) + self.g)*dt*dt # 3x1
        self.X = StateToChi(R_pred, v_pred, p_pred)

    def CorrectionDvl(self, y_dvl):
        H = np.zeros((3,9))
        H[:,3:6] = -np.eye(3)
        N_dvl = 0.5*np.eye(3)
        
        P_v = self.uuv.R_dvl_imu @ N_dvl @ self.uuv.R_dvl_imu.T + self.uuv.t_dvl_imu_skew @ self.Q_omega @ self.uuv.t_dvl_imu_skew.T
        z = np.array([[y_dvl[0], y_dvl[1], y_dvl[2], 1, 0]]).T
        v = (self.X @ z)[:3] # 3x1
        R = self.X[:3, :3]
        S = H @ self.P @ H.T + R @ P_v @ R.T
        S_inv = np.linalg.pinv(S)
        K = self.P @ H.T @ S_inv # 9x3
        Chi = StateVecToChi(K @ v)
        self.X = expm(Chi) @ self.X
        K_2 = np.eye(9) - K @ H
        self.P = K_2 @ self.P 
 
    
    def CorrectionDepth(self, y_depth):
        pos = np.array([[self.X[0, 4], self.X[1, 4], y_depth]]).T
        R = self.X[:3, :3]
        pos = -R.T @ pos
        y_depth = np.array([[pos[0], pos[1], pos[2], 0, 1]]).T
        
        N_D = block_diag(10*np.eye(3), np.zeros((2,2)))
        H = np.zeros((5,9))
        H[:3,6:] = -np.eye(3)
        b = np.array([[0, 0, 0, 0, 1]]).T
        
        S = H @ self.P @ H.T + N_D
        K = self.P @ H.T @ pinv(S)
        z = self.X @ y_depth - b   
        Chi = StateVecToChi(K@z) 
        self.X = self.X @ expm(Chi)
        K_2 = np.eye(9) - K @ H
        self.P = K_2 @ self.P
        
        
    def CorrectionCam(self, R_prev, R_in):
        R_in = self.uuv.R_cam_imu @ R_in @ self.uuv.R_cam_imu @ R_prev
               
        H = np.zeros((5,9))
        H[:3, :3] = SkewMatrix(np.array([[1, 1, 1]]).T) 
        N_cam = block_diag(0.1*np.eye(3), np.zeros((2,2)))
        b_cam = np.array([[1, 1, 1, 0, 0]]).T
        
        y_cam = np.zeros((5,1))
        y_cam[0,0] = R_in[0,0] + R_in[1,0] + R_in[2,0]
        y_cam[1,0] = R_in[0,1] + R_in[1,1] + R_in[2,1]
        y_cam[2,0] = R_in[0,2] + R_in[1,2] + R_in[2,2]
        
        S = H @ self.P @ H.T + N_cam
        K = self.P @ H.T @ pinv(S)
        z = self.X @ y_cam - b_cam   
        Chi = StateVecToChi(K@z) 
        self.X = self.X @ expm(Chi)
        K_2 = np.eye(9) - K @ H
        self.P = K_2 @ self.P
        
        

        



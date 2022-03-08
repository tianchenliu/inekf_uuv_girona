import numpy as np
from localization.utils import SkewMatrixVec

class Uuv_Girona:
    def __init__(self):      
        self.t_dvl_imu = np.array([0.0, 0.38, 0.07]).T
        self.t_dvl_imu_skew = SkewMatrixVec(self.t_dvl_imu)
        self.R_dvl_imu = np.array([ [0, 1, 0],
                                    [1, 0, 0],
                                    [0, 0, -1] ], dtype=float)
        
        self.t_ID = np.array([[-0.38, 0.0, 0.07]]).T
        self.t_ID_skew = SkewMatrixVec(self.t_dvl_imu)
        self.R_ID = np.array([[0, 1, 0],
                              [1, 0, 0],
                              [0, 0, -1] ], dtype=float)
        self.r_vec_ID = np.array([np.pi, 0, np.pi/2])
        
        self.R_depth_imu = np.array([ [0, 1, 0],
                                    [1, 0, 0],
                                    [0, 0, -1] ], dtype=float)
        self.t_depth_imu_z = -0.17

        self.t_cam_imu = np.array([0.0, 0.64, 0.09]).T
        self.t_cam_imu_skew = SkewMatrixVec(self.t_cam_imu)
        self.r_vec_imu_cam = np.array([np.pi, 0, 0])
        self.R_cam_imu = np.array([ [1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, -1] ], dtype=float)
        
        self.R_odom = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
     


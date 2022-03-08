import numpy as np
from numpy.linalg import norm, inv, pinv
from numpy import cos, sin
from scipy.linalg import logm, expm, block_diag
from scipy.spatial.transform import Rotation as ScipyRotation


def SkewMatrix(vec):
    '''
    Input: vec: 3x1 vector
    Output: vec_cross: 3x3 matrix
    '''
    vec_cross = np.array([  [0, -vec[2,0], vec[1,0]],
                    [vec[2,0], 0, -vec[0,0]],
                    [-vec[1,0], vec[0,0], 0]])
    return vec_cross

def SkewMatrixVec(vec):
    vec_cross = np.array([[ 0, -vec[2],  vec[1]],
                     [ vec[2],  0, -vec[0]],
                     [-vec[1],  vec[0], 0]])
    return vec_cross

def GetRotMatFromVec(vec):
    R = ScipyRotation.from_euler('xyz', vec).as_dcm()
    return R

def GetRotMatFromQuat(q):
    #R = ScipyRotation.from_quat(q).as_dcm()
    R = ScipyRotation.from_quat(q).as_matrix()
    return R

def ExpSO3(vec):
    a = np.linalg.norm(vec)
    if (a < 1e-8):
        R = np.eye(3)
    else:
        vec_cross = SkewMatrix(vec)
        R = np.eye(3) + (np.sin(a)/a)*vec_cross + ((1-np.cos(a))/(a*a))*np.matmul(vec_cross, vec_cross)
    return R

def AdjointSE2_3(X):
    '''
    Adjoint(X) for X in SE_2(3)
    '''
    R = X[:3,:3]
    v = np.array([X[:3, 3]]).T
    p = np.array([X[:3, 4]]).T

    Adj = np.zeros((9, 9))
    Adj[:3,:3] = R
    Adj[3:6, 3:6] = R
    Adj[6:, 6:] = R
    Adj[3:6,:3] = np.matmul(SkewMatrix(v), R)
    Adj[6:,:3] = np.matmul(SkewMatrix(p), R)

    return Adj


def StateToChi(R, v, p, m = None):
    # m: 3 x n_landmarks array
    if (m == None):
        X = np.eye(5)
    else:
        X = np.eye(5 + m.shape[1])
        X[:3, 5:] = m

    X[:3, :3] = R
    X[:3, 3:4] = v
    X[:3, 4:5] = p

    return X

def StateVecToChi(x):
    X = np.array([[0, -x[2], x[1], x[3], x[6]],
                [x[2], 0, -x[0], x[4], x[7]],
                [-x[1], x[0], 0, x[5], x[8]],
                [0,0,0,0,0],
                [0,0,0,0,0]], dtype=float)
    return X

def ChiToState(X):
    R = X[:3, :3]
    v = X[:3, 3:4]
    p = X[:3, 4:5]
    if (X.shape[1] > 5):
        m = X[:3, 5:]
    else:
        m = None

    return R, v, p, m

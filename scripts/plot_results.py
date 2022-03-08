#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from data_reader import Girona

def PlotPosition(t1, p1, t2, p2, save_filename):
    plt.rcParams.update({'font.size': 18})

    fig, axes = plt.subplots(3, sharey=False)
    #fig.suptitle('Underwater Vehicle Trajectory')
    fig.supxlabel('t (s)')

    
    axes[0].set_ylabel('x (m)')
    axes[1].set_ylabel('y (m)')
    axes[2].set_ylabel('z (m)')

    axes[0].plot(t1[:], p1[:, 0], label="odometry")
    axes[0].plot(t2[:], p2[:, 0], 'tab:red', label="predict")

    axes[1].plot(t1[:], p1[:, 1], label="odometry")
    axes[1].plot(t2[:], p2[:, 1], 'tab:red', label="predict")

    axes[2].plot(t1[:], p1[:, 2], label="odometry")
    axes[2].plot(t2[:], p2[:, 2], 'tab:red', label="predict")
    axes[-1].legend(loc='best')

    fig.align_ylabels(axes[:])
    
    f = plt.gcf()
    f.set_size_inches(12, 8)
    #plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plt.savefig(save_filename, dpi=300)
    #plt.show()

def PlotSingle(data, data_file, result_fig):
    all_X_gt = data.y_odom.T
    all_time_gt = data.ts_odom[0, :] - data.ts_odom[0, 0]
    p_gt = (data.uuv.R_odom @ data.y_odom_p).T    

    filter_data = np.load(data_file)
    ts_pred = filter_data['time_pred'] - filter_data['time_pred'][0]
    X_pred = filter_data['X_pred'][:, :3, 4]
    
    PlotPosition(all_time_gt, p_gt, ts_pred, X_pred, result_fig)

def PlotMulti(data):
    t1 = data.ts_odom[0, :] - data.ts_odom[0, 0]
    p1 = (data.uuv.R_odom @ data.y_odom_p).T    

    filter_data = np.load('./results/girona_idpc_multi.npz')
    all_t = filter_data['all_t']
    all_X = filter_data['all_X']
    
    t2 = all_t[0] - all_t[0][0]
    
    plt.rcParams.update({'font.size': 18})

    fig, axes = plt.subplots(3, sharey=False)
    fig.supxlabel('t (s)')
    
    axes[0].set_ylabel('x (m)')
    axes[1].set_ylabel('y (m)')
    axes[2].set_ylabel('z (m)')
    
    axes[0].plot(t1[:], p1[:, 0], label="odometry")
    axes[1].plot(t1[:], p1[:, 1], label="odometry")
    axes[2].plot(t1[:], p1[:, 2], label="odometry")
    
    for i in range(5):
        p2 = all_X[i, :,:3, 4]
        axes[0].plot(t2[:], p2[:, 0], 'tab:red', label="predict")
        axes[1].plot(t2[:], p2[:, 1], 'tab:red', label="predict")
        axes[2].plot(t2[:], p2[:, 2], 'tab:red', label="predict")
        
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='best')
    
    fig.align_ylabels(axes[:])
    
    f = plt.gcf()
    f.set_size_inches(12, 8)

    plt.savefig("./results/girona_idpc_multi.png", dpi=300)
    #plt.show()


if __name__ == "__main__": 
    data = Girona()
    data.GetData()
    
    #data_file = './results/girona_idp_one.npz'
    #output_fig = './results/girona_idp_one.png'
    #PlotSingle(data, data_file, output_fig)
    
    #data_file = './results/girona_idpc_one.npz'
    #output_fig = './results/girona_idpc_one.png'
    #PlotSingle(data, data_file, output_fig)
    
    PlotMulti(data)
    

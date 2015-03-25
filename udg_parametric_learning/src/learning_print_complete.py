#!/usr/bin/env python

import numpy as np
import math

import matplotlib.pyplot as plt


def load_trajectory(file_name, samples):
    """
    Load Trajectory from the last point to the beginning
    """
    print 'Loading Trajectory ' + file_name + ' :'
    demonstrations = []
    if len(samples) != 0:
        for n in xrange(len(samples)):
            #print 'Loading Demonstration ' + file_name + "_" + str(ni)
            ni = samples[n]
            if type(file_name) is str:
                logfile = open(file_name + "_" + str(ni) + ".csv",
                               "r").readlines()
            else:
                #The file name is a list of elements
                logfile = open(file_name[n] + "_" + str(ni) + ".csv",
                               "r").readlines()
                # vars = np.zeros((1, self.nbVar))
                # Added the time to the var
            data_demo = np.array([[]])
            for line in logfile:
                if len(data_demo[0]) == 0:
                    data_demo = np.array([line.split()], dtype=np.float64)
                else:
                    data_demo = np.append(
                        data_demo,
                        np.array([line.split()], dtype=np.float64),
                        axis=0)
            demonstrations.append(data_demo)
    else:
        logfile = open(file_name + ".csv", "r").readlines()
        data_demo = np.array([[]])
        for line in logfile:
            if len(data_demo[0]) == 0:
                data_demo = np.array([line.split()], dtype=np.float64)
            else:
                data_demo = np.append(
                    data_demo,
                    np.array([line.split()], dtype=np.float64),
                    axis=0)
        demonstrations.append(data_demo)

    return demonstrations

if __name__ == '__main__':
    demos_group_1 = load_trajectory(
        '../parametric_data/trajectory_demonstration', [0,1])
    demos_group_2 = load_trajectory(
        '../parametric_data/trajectory_demonstration', [3,7])

    traj = load_trajectory(
        '../parametric_data/trajectory_played_sim',[])

    #Plot values
    plt.ion()
    f, axis = plt.subplots(4, sharex=True)
    f.suptitle("AUV")
    for i in xrange(len(demos_group_1)):
        #plot time, x
        axis[0].plot(demos_group_1[i][:,0] - demos_group_1[i][1,0], demos_group_1[i][:,1], color='b')
        #plot time, y
        axis[1].plot(demos_group_1[i][:,0] - demos_group_1[i][1,0], demos_group_1[i][:,2], color='b')
        #plot time, z
        axis[2].plot(demos_group_1[i][:,0] - demos_group_1[i][1,0], demos_group_1[i][:,3], color='b')
        #plot time, yaw
        axis[3].plot(demos_group_1[i][:,0] - demos_group_1[i][1,0], demos_group_1[i][:,4], color='b')

    for i in xrange(len(demos_group_2)):
        #plot time, x
        axis[0].plot(demos_group_2[i][:,0] - demos_group_2[i][1,0], demos_group_2[i][:,1], color='r')
        #plot time, y
        axis[1].plot(demos_group_2[i][:,0] - demos_group_2[i][1,0], demos_group_2[i][:,2], color='r')
        #plot time, z
        axis[2].plot(demos_group_2[i][:,0] - demos_group_2[i][1,0], demos_group_2[i][:,3], color='r')
        #plot time, yaw
        axis[3].plot(demos_group_2[i][:,0] - demos_group_2[i][1,0], demos_group_2[i][:,4], color='r')

    for i in xrange(len(traj)):
        #plot time, x
        axis[0].plot(traj[i][:,0] - traj[i][1,0], traj[i][:,1], color='g')
        #plot time, y
        axis[1].plot(traj[i][:,0] - traj[i][1,0], traj[i][:,2], color='g')
        #plot time, z
        axis[2].plot(traj[i][:,0] - traj[i][1,0], traj[i][:,3], color='g')
        #plot time, yaw
        axis[3].plot(traj[i][:,0] - traj[i][1,0], traj[i][:,4], color='g')

    # for i in xrange(len(traj)):
    #     plt.plot(traj[i][:,0] - traj[i][1,0], traj[i][:,1], color='r')
    plt.ioff()
    plt.show()

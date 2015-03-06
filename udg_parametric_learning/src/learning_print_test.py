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
    demos = load_trajectory(
        './fake_trajectory', [1,2])
    traj = load_trajectory(
        'simulated_trajectory_param',[])

    #Plot values
    plt.ion()
#    f, axis = plt.subplots(1, sharex=True)
#    f.suptitle("1D")

    for i in xrange(len(demos)):
        #plot time, x
        plt.plot(demos[i][:,0] - demos[i][1,0], demos[i][:,1], color='b')

    for i in xrange(len(traj)):
        plt.plot(traj[i][:,0] - traj[i][1,0], traj[i][:,1], color='r')
    plt.ioff()
    plt.show()

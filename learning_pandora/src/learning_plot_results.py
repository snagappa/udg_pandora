#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def helloWorld():
    print 'Hello World'

def load_trajectory(file_name, samples):
    """
    Load Trajectory from the last point to the beginning
    """
    print 'Loading Trajectory ' + file_name + ' :'
    demonstrations = []
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
    return demonstrations

if __name__ == '__main__':
    #load demonstration
    demos = load_trajectory('../learning_data/trajectory_demonstration', [0, 1, 2])
    #simulation = load_trajectory('../config/trajectoryPlayed_individual', [0])

    print 'Size ' + str(len(demos))
    #Plot values
    plt.ion()
    f, axis = plt.subplots(4, sharex=True)
    for i in xrange(len(demos)):
        #plot time, x
        axis[0].plot(demos[i][:,10] - demos[i][1,10], demos[i][:,0], color='b')
        #plot time, x
        axis[1].plot(demos[i][:,10] - demos[i][1,10], demos[i][:,1], color='b')
        #plot time, x
        axis[2].plot(demos[i][:,10] - demos[i][1,10], demos[i][:,2], color='b')
        #plot time, x
        axis[3].plot(demos[i][:,10] - demos[i][1,10], demos[i][:,3], color='b')
    plt.show()

    f, axis = plt.subplots(4, sharex=True)
    for i in xrange(len(demos)):
        #plot time, x
        axis[0].plot(demos[i][:,10] - demos[i][1,10], demos[i][:,4], color='b')
        #plot time, x
        axis[1].plot(demos[i][:,10] - demos[i][1,10], demos[i][:,5], color='b')
        #plot time, x
        axis[2].plot(demos[i][:,10] - demos[i][1,10], demos[i][:,6], color='b')
        #plot time, x
        axis[3].plot(demos[i][:,10] - demos[i][1,10], demos[i][:,9], color='b')
    plt.ioff()
    plt.show()
#    print 'Shape of the sensor ' + str(demos[0][0,0])

import numpy as np
import matplotlib.pyplot as plt

def load_trajectory(file_name, samples):
    """
    Load Trajectory from the last point to the beginning
    """



if __name__ == '__main__':
    demos = True
    sim = False
    real = True
    #load demonstration
    if demos:
        demos = load_trajectory('../learning_data/trajectory_demonstration_v3',
                                [70, 71, 72]) #[16,17]) [19,21])
    if sim:
        simulation = load_trajectory('../learning_data/trajectoryPlayed_individual',
                                     [])
    if real:
        real = load_trajectory('../learning_data/real_traj',
                               [])
    #Plot values
    plt.ion()
    f, axis = plt.subplots(4, sharex=True)
    f.suptitle("AUV")
    #plot Demos
    if demos:
        for i in xrange(len(demos)):
        #plot time, x
            axis[0].plot(demos[i][:,0] - demos[i][1,0], demos[i][:,1], color='b')
        #plot time, y
            axis[1].plot(demos[i][:,0] - demos[i][1,0], demos[i][:,2], color='b')
        #plot time, z
            axis[2].plot(demos[i][:,0] - demos[i][1,0], demos[i][:,3], color='b')
        #plot time, yaw
            axis[3].plot(demos[i][:,0] - demos[i][1,0], demos[i][:,4], color='b')
    #plot reproduction simulated
    if sim:
        for i in xrange(len(simulation)):
        #plot time, x
            axis[0].plot(simulation[i][:,0] - simulation[i][1,0],
                         simulation[i][:,1], color='r')
        #plot time, y
            axis[1].plot(simulation[i][:,0] - simulation[i][1,0],
                         simulation[i][:,2], color='r')
        #plot time, z
            axis[2].plot(simulation[i][:,0] - simulation[i][1,0],
                         simulation[i][:,3], color='r')
        #plot time, yaw
            axis[3].plot(simulation[i][:,0] - simulation[i][1,0],
                         simulation[i][:,4], color='r')
    #plot reproductions
    if real:
        for i in xrange(len(real)):
        #plot time, x
            axis[0].plot(real[i][:,0] - real[i][1,0],
                         real[i][:,1], color='g')
        #plot time, y
            axis[1].plot(real[i][:,0] - real[i][1,0],
                         real[i][:,2], color='g')
        #plot time, z
            axis[2].plot(real[i][:,0] - real[i][1,0],
                         real[i][:,3], color='g')
        #plot time, yaw
            axis[3].plot(real[i][:,0] - real[i][1,0],
                         real[i][:,4], color='g')
    plt.show()

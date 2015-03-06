#!/usr/bin/env python

import numpy as np
import math

from learning_dmp_parametric import LearningDmpParametric
from learning_dmp_param_reproductor import LearningDmpParamReproductor

#Learning the Parametric models
kP = -99.0
kV = -99.0
kPmin = 1
kPmax = 0.1
alpha = 1.0
nbStates = 3
dof_list = [1]
nbData = 400
demonstration_file = 'fake_trajectory'
demonstrations = [1,2]
init_time = [0.0, 0.0]
end_time = [119.8, 119.8]
param_value = [0,1]
export_filename = 'model_parametric'

dmp_p = LearningDmpParametric(kP, kV, kPmin, kPmax,
                              alpha, nbStates, dof_list,
                              nbData, demonstration_file,
                              demonstrations,
                              init_time, end_time,
                              param_value, export_filename)
dmp_p.trainningDMP()
dmp_p.exportPlayData()

file_name = 'model_parametric'
file_path = '.'
alpha = 1.0
dof_list = [1]
dof = len(dof_list)
interval_time = 0.2
nb_groups = 2

reproductor = LearningDmpParamReproductor(
    file_name,
    file_path,
    dof,
    alpha,
    interval_time,
    nb_groups)

current_pose = [0.0]
current_vel = [0.0]
action = 1.0
param = 0.0

file_trajectory = open('simulated_trajectory_param.csv', 'w')
time = 0.0

while current_pose and current_vel:
    s = str(time) + ' ' + str(current_pose[0]) + '\n'
    file_trajectory.write(s)
    [current_pose, current_vel] = reproductor.generateNewPose(
        current_pose,
        current_vel,
        action,
        param)
    time += interval_time

print 'Simulation finished, check the File'

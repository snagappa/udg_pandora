#!/usr/bin/env python

from learning_dmp_parametric import LearningDmpParametric

kP = [[-99.0,-99.0], [-99.0,-99.0], [-99.0,-99.0], [-99.0,-99.0]]
kV = [[-99.0,-99.0], [-99.0, -99.0], [-99.0,-99.0], [-99.0,-99.0]]
# kPmax = [[30.0,25.0],[20.0,10.0]]
# kPmin = [[10.0, 10.0], [1.0, 0.1]]
# kPmax = [[1.3,1.5],[1.5,1.5],[1.0,1.0],[1.5,1.0]]
# kPmin = [[1.0, 1.2], [1.2, 1.2],[0.5,0.5],[1.0,0.5]]
kPmax = [[1.3,1.5],[1.5,1.5],[1.5,1.0],[1.5,1.0]]
kPmin = [[1.0, 1.2], [1.2, 1.2],[1.0,0.5],[1.0,0.5]]
alpha = 1.0
#nbStates = [[8,8],[11,11],[8,8],[11,11]]
nbStates = [[8,8],[11,11],[8,8],[11,11]]
#dof_list = [1]
nbData = 400
demonstration_file = '../parametric_data/trajectory_demonstration'
demonstrations = [0,1,3,15,16,17,19,20]
#demonstrations = [0,1,3,40,41,42,43,44,45]

init_time = [1428588977.298196, 1428589173.029574, 1428589584.775652, 1428592679.723152, 1428593546.091038, 1429021000.577938, 1429021510.077681, 1429022157.790316]
end_time = [1428589069.662532, 1428589252.004525, 1428589670.221962, 1428592774.837879, 1428593625.838648, 1429021101.971818, 1429021587.908463, 1429022249.372202]

# init_time = [1428588977.298196, 1428589173.029574, 1428589584.775652, 1429180932.8103, 1429181890.07992, 1429182602.231376, 1429183162.824294, 1429183398.21784, 1429183626.841842]
# end_time = [1428589069.662532, 1428589252.004525, 1428589670.221962, 1429181027.309299, 1429181979.911701, 1429182707.584216, 1429183250.672627, 1429183495.382147, 1429183720.951444]

param_value = [0,0,0,1,1,1,1,1]
param_samples = [3,5]
# param_value = [0,0,0,1,1,1,1,1,1]
# param_samples = [3,6]

export_filename = [
    '../parametric_data/learned_data_complete_individual_short_auv_x',
    '../parametric_data/learned_data_complete_individual_short_auv_z',
    '../parametric_data/learned_data_complete_individual_short_ee_x',
    '../parametric_data/learned_data_complete_individual_short_ee_z']

# AUV X Y Yaw
dof_list = [1,1,0,1,0,0,0,0,0,0]
dmp_1 = LearningDmpParametric(kP[0], kV[0], kPmin[0], kPmax[0],
                              alpha, nbStates[0], dof_list,
                              nbData, demonstration_file,
                              demonstrations,
                              init_time, end_time, param_value,
                              param_samples, export_filename[0])
dmp_1.trainningDMP()
dmp_1.exportPlayData()

# AUV Z
dof_list = [0,0,1,0,0,0,0,0,0,0]
dmp_1 = LearningDmpParametric(kP[1], kV[1], kPmin[1], kPmax[1],
                              alpha, nbStates[1], dof_list,
                              nbData, demonstration_file,
                              demonstrations,
                              init_time, end_time, param_value,
                              param_samples, export_filename[1])
dmp_1.trainningDMP()
dmp_1.exportPlayData()


# EE X and Y
dof_list = [0,0,0,0,1,1,0,0,0,1]
dmp_1 = LearningDmpParametric(kP[2], kV[2], kPmin[2], kPmax[2],
                              alpha, nbStates[2], dof_list,
                              nbData, demonstration_file,
                              demonstrations,
                              init_time, end_time, param_value,
                              param_samples, export_filename[2])
dmp_1.trainningDMP()
dmp_1.exportPlayData()

# # EE Z
dof_list = [0,0,0,0,0,0,1,0,0,0]
dmp_1 = LearningDmpParametric(kP[3], kV[3], kPmin[3], kPmax[3],
                              alpha, nbStates[3], dof_list,
                              nbData, demonstration_file,
                              demonstrations,
                              init_time, end_time, param_value,
                              param_samples, export_filename[3])
dmp_1.trainningDMP()
dmp_1.exportPlayData()

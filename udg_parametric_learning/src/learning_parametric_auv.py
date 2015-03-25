#!/usr/bin/env python

from learning_dmp_parametric import LearningDmpParametric

kP = [[-99.0,-99.0], [-99.0,-99.0]]
kV = [[-99.0,-99.0], [-99.0, -99.0]]
kPmax = [[3.0,10.0],[3.0,10.0]]
kPmin = [[0.1, 0.1], [0.1, 0.1]]
alpha = 1.0
nbStates = [[5,10],[5,10]]
#dof_list = [1]
nbData = 400
demonstration_file = '../parametric_data/trajectory_demonstration'
demonstrations = [0,1,3,7]
init_time = [1427198369.784869, 1427198536.944197,1427207200.301873,1427209602.339454]
end_time = [1427198445.570745, 1427198613.754822,1427207306.514012,1427209708.369823]
param_value = [0,0,1,1]
param_samples = [2,2]
export_filename = [
    '../parametric_data/learned_data_complete_individual_short_auv_x',
    '../parametric_data/learned_data_complete_individual_short_auv_z']
#, 'learned_data_complete_individual_short_ee_x.txt', 'learned_data_complete_individual_short_ee_z.txt']

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
#dof_list = [0,0,0,0,1,1,0,1,0,0]
# dof_list = [0,0,0,0,1,1,0,0,0,1]
# dmp_1 = LearningDmpGeneric(self.kP[2], self.kV[2], self.kPmin[2], self.kPmax[2],
#                            self.alpha, self.nbStates[2], dof_list,
#                            self.nbData, self.demonstration_file,
#                            self.demonstrations,
#                            init_time, end_time, self.export_filename[2])
# dmp_1.trainningDMP()
# dmp_1.exportPlayData()

# # EE Z
# dof_list = [0,0,0,0,0,0,1,0,0,0]
# dmp_1 = LearningDmpGeneric(self.kP[3], self.kV[3], self.kPmin[3], self.kPmax[3],
#                            self.alpha, self.nbStates[3], dof_list,
#                            self.nbData, self.demonstration_file,
#                            self.demonstrations,
#                            init_time, end_time, self.export_filename[3])
# dmp_1.trainningDMP()
# dmp_1.exportPlayData()

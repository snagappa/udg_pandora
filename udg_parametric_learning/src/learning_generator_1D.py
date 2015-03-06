#!/usr/bin/env python

import numpy as np
import math

#generate first demo

t = np.arange(0,120,0.2)
y = np.ones(len(t))*1
y2 = np.copy(y)

t_obs_up = np.arange(20,40,0.2)
t_obs_down = np.arange(60,80,0.2)

index = np.where(t==20)[0][0]
for i in range(len(t_obs_up)):
    y2[index+i] += t_obs_up[i] - t_obs_up[0]

init = int(40/0.2)
end = int(60/0.2)
y2[init:end] = 21

aux = 21
index = np.where(t==60)[0][0]
for i in range(len(t_obs_down)):
    aux -= 0.2
    y2[index+i] = aux

file_1 = open('fake_trajectory_1.csv', 'w')
file_2 = open('fake_trajectory_2.csv', 'w')

for i in range(len(t)):
    s = str(t[i]) + ' ' + str(y[i]) + '\n'
    file_1.write(s)
    s = str(t[i]) + ' ' + str(y2[i]) + '\n'
    file_2.write(s)

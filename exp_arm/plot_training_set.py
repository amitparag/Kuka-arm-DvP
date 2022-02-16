
# This files contains handy scripts to test the performance and debug the learned value function
# Author : Sébastien Kleff
# Date : 09/20/2021

import numpy as np
from utils import path_utils, ocp_utils, plot_utils, pin_utils
from pinocchio.robot_wrapper import RobotWrapper
import torch
import sys
np.set_printoptions(precision=4, linewidth=180)
import matplotlib.pyplot as plt
import os
from datagen import samples_uniform_mixed_adaptive

# Load robot and OCP config
robot = RobotWrapper.BuildFromURDF(path_utils.kuka_urdf_path(), path_utils.kuka_mesh_path())
config = path_utils.load_config_file('static_reaching_task_ocp2')
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
robot.framesForwardKinematics(q0)
robot.computeJointJacobians(q0)
nq=robot.model.nq; nv=robot.model.nv; nu=nq; nx=nq+nv
N_h = config['N_h']
dt = config['dt']
id_ee = robot.model.getFrameId('contact')


N_SAMPLES = 400
DDPS_DATA = []
samples   =   samples_uniform_mixed_adaptive(nb_samples=N_SAMPLES) 
# Init and solve
for k,x in enumerate(samples):
    print("sample "+str(k)+"/"+str(len(samples)))
    robot.framesForwardKinematics( x[:nq])
    robot.computeJointJacobians( x[:nq])
    ddp = ocp_utils.init_DDP(robot, config, x, critic=None, 
                                    callbacks=False, 
                                    which_costs=config['WHICH_COSTS'],
                                    dt=dt, N_h=N_h) 
    ug = pin_utils.get_u_grav(x[:nq], robot)
    xs_init = [x for i in range(N_h+1)]
    us_init = [ug  for i in range(N_h)]
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    ddp_data = plot_utils.extract_ddp_data(ddp)
    DDPS_DATA.append(ddp_data)

fig, ax = plot_utils.plot_ddp_results(DDPS_DATA, SHOW=True, sampling_plot=6)
plot_utils.plot_refs(fig, ax, config)
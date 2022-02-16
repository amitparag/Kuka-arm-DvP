# This files contains the code to generate the video in Gepetto viewer
# testing the learned VF by generating trajectories croco(0..T)+VF 
# starting from random states
# Author : Sébastien Kleff
# Date : 09/21/2021

import numpy as np
from utils import path_utils, ocp_utils, plot_utils, pin_utils
import torch
np.set_printoptions(precision=4, linewidth=180)
import os
from datagen import samples_uniform_IK, samples_uniform
from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin
from utils import path_utils
import time
import random

# Load config
config = path_utils.load_config_file('static_reaching_task_ocp2')
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])  
N_h = config['N_h']
dt = config['dt'] 
# Load robot model + viewer
robot = RobotWrapper.BuildFromURDF(path_utils.kuka_urdf_path(), path_utils.kuka_mesh_path())
robot.initDisplay(loadModel=True); robot.display(q0)
viewer = robot.viz.viewer; gui = viewer.gui
nq=robot.model.nq; nv=robot.model.nv; nu=nq; nx=nq+nv
id_ee = robot.model.getFrameId('contact')


N=200
EPS_P = [0.05, 0.15, 0.25]
EPS_V = [0.005, 0.01, 0.015]
# Sample test points
samples = samples_uniform_IK(nb_samples=N, eps_p=EPS_P, eps_v=EPS_V)
random.shuffle(samples)

def visualize(samples, sleep=0.01):
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)
    robot.display(q0)
    M_des = robot.data.oMf[id_ee].copy()
    M_des.translation = np.asarray(config['p_des'])
    tf_des = pin.utils.se3ToXYZQUAT(M_des)
    if(not gui.nodeExists('world/p_des')):
        gui.addSphere('world/p_des', .02, [1. ,0 ,0, 1.])  
    gui.applyConfiguration('world/p_des', tf_des)
    colors = [[1., 0., 0., 0.6], [0., 1., 0., 0.3], [0., 0., 1., 0.1]]
    for i in [2, 1, 0]:
        gui.addBox('world/p_bounds_'+str(i),   2*EPS_P[i], 2*EPS_P[i], 2*EPS_P[i],  [1., 1./float(i+1), 1.-1./float(i+1), 0.3]) 
        gui.applyConfiguration('world/p_bounds_'+str(i), tf_des)
    for k,sample in enumerate(samples):
        # q = sample[:nq]
        robot.display(sample[:nq])
        # Update model and display sample
        robot.framesForwardKinematics(sample[:nq])
        robot.computeJointJacobians(sample[:nq])
        M_ = robot.data.oMf[id_ee]
        gui.addSphere('world/sample'+str(k), .01, [0. ,0 ,1., .8])  
        tf_ = pin.utils.se3ToXYZQUAT(M_)
        gui.applyConfiguration('world/sample'+str(k), tf_)
        gui.refresh()
        time.sleep(0.5)
    # Check samples


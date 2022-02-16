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
# Load trained NN
resultspath = path_utils.results_path()
path = os.path.join(resultspath, 'trained_models/dvp/Order_1/Horizon_200/eps_9.pth')
Net  = torch.load(path)

# Test the learned V.F.
DDPS_DATA =[]
WS = False
N=20
EPS_P = 0.3
# Sample test points
samples = samples_uniform_IK(nb_samples=N//2, eps_p=EPS_P, eps_v=0.0)
samples.extend(samples_uniform(nb_samples=N//2, eps_q=1., eps_v=0.))
random.shuffle(samples)
# samples = samples_uniform(nb_samples=N, eps_q=1., eps_v=0.)
# Ref for warm start
ddp_ref = ocp_utils.init_DDP(robot, config, x0, critic=None, callbacks=False, which_costs=config['WHICH_COSTS'], dt=dt, N_h=N_h)
# Solve for several samples 
for k,x in enumerate(samples):
    robot.framesForwardKinematics(x[:nq])
    robot.computeJointJacobians(x[:nq])
    ddp = ocp_utils.init_DDP(robot, config, x, critic=Net, 
                                    callbacks=False, 
                                    which_costs=config['WHICH_COSTS'],
                                    dt=dt, N_h=N_h) 
    ug = pin_utils.get_u_grav(x[:nq], robot)
    ddp_ref.problem.x0 = x
    ddp_ref.solve( [x for i in range(N_h+1)] , [ug  for i in range(N_h)], maxiter=config['maxiter'], isFeasible=False)
    # Warm start using the croco ref
    xs_init = [x for i in range(N_h+1)]# [ddp_ref.xs[i] for i in range(N_h+1)]
    us_init = [ug  for i in range(N_h)] #[ddp_ref.us[i]  for i in range(N_h)]
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
    # Solve for each sample and record
    ddp_data = plot_utils.extract_ddp_data(ddp)
    DDPS_DATA.append(ddp_data)

from test_trained import test_trained_single
DDPS_DATA = [test_trained_single(path, x0=x, PLOT=False, logs=False) for x in samples]
# Plot results
fig, ax = plot_utils.plot_ddp_results(DDPS_DATA, SHOW=False, sampling_plot=1)
plot_utils.plot_refs(fig, ax, config)
# remove legent p
# ax['p'].get_legend().remove()

def animate(data, sleep=dt):
    M_des = robot.data.oMf[id_ee].copy()
    M_des.translation = np.asarray(config['p_des'])
    tf_des = pin.utils.se3ToXYZQUAT(M_des)
    if(not gui.nodeExists('world/p_des')):
        gui.addSphere('world/p_des', .02, [1. ,0 ,0, 1.])  
    gui.applyConfiguration('world/p_des', tf_des)
    for k,d in enumerate(data):
        print("Sample "+str(k)+"/"+str(len(data)))
        q = np.array(d['xs'])[:,:nq]
        for i in range(N_h+1):
            robot.display(q[i])
            gui.refresh()
            time.sleep(sleep)

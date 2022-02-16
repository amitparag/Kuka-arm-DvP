# This files contains the code to debug issues with learned VF
# Solve OCP from a given x0 using
#  - Croco on long horizon (ddp1)
#  - Croco short horizon + learned VF (ddp2) using ddp1 as warm-start
#  - Croco short horizon (ddp3)
# starting from random states
# Author : Sébastien Kleff
# Date : 09/21/2021

import numpy as np
from utils import path_utils, ocp_utils, plot_utils, pin_utils
import torch
np.set_printoptions(precision=4, linewidth=180)
import matplotlib.pyplot as plt
import os
from pinocchio.robot_wrapper import RobotWrapper
from utils import path_utils


config = path_utils.load_config_file('static_reaching_task_ocp2')
robot = RobotWrapper.BuildFromURDF(path_utils.kuka_urdf_path(), path_utils.kuka_mesh_path())
nq=robot.model.nq; nv=robot.model.nv; nu=nq; nx=nq+nv
x0 = np.array([ 0.6934,  0.935 ,  1.3379, -0.9354,  1.5454,  1.8671,  2.0287,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ])
q0 = x0[:nq]
v0 = x0[nv:]

robot.initDisplay(loadModel=True)
N_h = config['N_h']
dt = config['dt']
id_ee = robot.model.getFrameId('contact')
resultspath = path_utils.results_path()
path = os.path.join(resultspath, 'trained_models/dvp/Order_1/Horizon_200/eps_9.pth')



def test_trained_single(critic_path, PLOT=False, x0=x0, logs=True):
    """
    Solve an OCP using the trained NN as a terminal cost
    """
    # Load trained NN 
    Net  = torch.load(critic_path)
    # Init and solve
    q0 = x0[:nq]
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)
    ddp = ocp_utils.init_DDP(robot, config, x0, critic=Net, 
                                   callbacks=logs, 
                                   which_costs=config['WHICH_COSTS'],
                                   dt=dt, N_h=N_h) 
    ug = pin_utils.get_u_grav(q0, robot)
    xs_init = [x0 for i in range(N_h+1)]
    us_init = [ug  for i in range(N_h)]
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    ddp_data = plot_utils.extract_ddp_data(ddp)
    # Plot
    if(PLOT):
        fig, ax = plot_utils.plot_ddp_results(ddp_data, SHOW=False)
        plot_utils.plot_refs(fig, ax, config)
    return ddp_data

def test_trained_multiple(critic_path, N=20, PLOT=False):
    """
    Solve N OCPs using the trained NN as a terminal cost
    from sampled test points x0
    """
    # Sample test points
    samples   =   samples_uniform(nb_samples=N)
    # Solve for each sample and record
    DDPS_DATA = [test_trained_single(critic_path, x0=x, PLOT=False, logs=False) for x in samples]
    # Plot results
    if(PLOT):
        fig, ax = plot_utils.plot_ddp_results(DDPS_DATA, SHOW=False, sampling_plot=1)
        plot_utils.plot_refs(fig, ax, config)
    return DDPS_DATA, samples

iter_number=5
# Solve OCP over [0,...,(k+1)T] using Crocoddyl
ddp1 = ocp_utils.init_DDP(robot, config, x0, critic=None, 
                                callbacks=False, 
                                which_costs=config['WHICH_COSTS'],
                                dt = dt, N_h=(iter_number+1)*N_h) 
ug = pin_utils.get_u_grav(q0, robot)
xs_init = [x0 for i in range((iter_number+1)*N_h+1)]
us_init = [ug  for i in range((iter_number+1)*N_h)]

# Solve OCP over [0,...,T] using Crocoddyl + learned VF
ddp1.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
Net = torch.load(path)
ddp2 = ocp_utils.init_DDP(robot, config, x0, critic=Net,
                                callbacks=False, 
                                which_costs=config['WHICH_COSTS'],
                                dt = dt, N_h=N_h) 
# Warm start using the croco ref
xs_init = [ddp1.xs[i] for i in range(N_h+1)]
us_init = [ddp1.us[i]  for i in range(N_h)]
ddp2.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)

 
# Plot
d1 = plot_utils.extract_ddp_data(ddp1)
d2 = plot_utils.extract_ddp_data(ddp2)
label1 ='OCP([0,...,'+ str(iter_number+1)+'T])'
label2='OCP([0,...,T]) + V_'+str(iter_number)+' ( warm-started from OCP([0,...,'+str(iter_number+1)+'T]) )'

# Add plot of Croco [0,..T] to compare
label3='OCP([0,...,T])'
ddp3 = ocp_utils.init_DDP(robot, config, x0, critic=None, callbacks=True, which_costs=config['WHICH_COSTS'], dt=dt, N_h=N_h)
ug = pin_utils.get_u_grav(q0, robot)
xs_init = [x0 for i in range(N_h+1)]
us_init = [ug  for i in range(N_h)]
ddp3.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
d3 = plot_utils.extract_ddp_data(ddp3)

# Plot stuff
fig, ax = plot_utils.plot_ddp_results([d1, d2, d3], labels=[label1, label2, label3], SHOW=False, marker='o', sampling_plot=1)
plot_utils.plot_refs(fig, ax, config, SHOW=True)
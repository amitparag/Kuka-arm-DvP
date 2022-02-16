
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
from datagen import samples_uniform_IK

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
resultspath = path_utils.results_path()

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
    samples   =   samples_uniform_IK(nb_samples=N, eps_p=0.05, eps_v=0.01)
    # Solve for each sample and record
    DDPS_DATA = [test_trained_single(critic_path, x0=x, PLOT=False, logs=False) for x in samples]
    # Plot results
    if(PLOT):
        fig, ax = plot_utils.plot_ddp_results(DDPS_DATA, SHOW=False, sampling_plot=1)
        plot_utils.plot_refs(fig, ax, config)
    return DDPS_DATA


def check_bellman(horizon=200, iter_number=1, WARM_START=0, PLOT=True):
    """
    Check that recursive property still holds on trained model: 
         - solve using croco over [0,..,(k+1)T]
         - solve using croco over [0,..,T]  + Vk
     where k = iter_number > 0 is the iteration number of the trained NN to be checked, i.e.
     when iter_number=1, we use eps_0.pth (a.k.a "V_1")
     when iter_number=2, we use eps_1.pth (a.k.a "V_2")
     ... 
     Should be the same  
    """
    # Solve OCP over [0,...,(k+1)T] using Crocoddyl
    ddp1 = ocp_utils.init_DDP(robot, config, x0, critic=None, 
                                    callbacks=False, 
                                    which_costs=config['WHICH_COSTS'],
                                    dt = dt, N_h=(iter_number+1)*N_h) 
    ug = pin_utils.get_u_grav(q0, robot)
    xs_init = [x0 for i in range((iter_number+1)*N_h+1)]
    us_init = [ug  for i in range((iter_number+1)*N_h)]
    # Solve
    ddp1.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    print("WITHOUT CRITIC : Croco([0,...,"+ str(iter_number+1)+"T])")
    print("   Cost     = ", ddp1.cost)
    print("   V(xT)    = ", ddp1.problem.runningDatas[N_h].cost)
    print("   V_x(xT)  = \n", ddp1.problem.runningDatas[N_h].Lx)
    print("   V_xx(xT) = \n", ddp1.problem.runningDatas[N_h].Lxx)
    print("\n")

    # Solve OCP over [0,...,T] using k^th trained NN estimate as terminal model
    critic_path = os.path.join(resultspath, f"trained_models/dvp/Order_{1}/Horizon_{horizon}/")
    critic_name = os.path.join(critic_path, "eps_"+str(iter_number-1)+".pth")
    print("Selecting trained network : eps_"+str(iter_number-1)+".pth\n")
    Net = torch.load(critic_name)
    ddp2 = ocp_utils.init_DDP(robot, config, x0, critic=Net,
                                    callbacks=False, 
                                    which_costs=config['WHICH_COSTS'],
                                    dt = dt, N_h=N_h) 
    if(bool(WARM_START)):
        # Warm start using the croco ref
        xs_init = [ddp1.xs[i] for i in range(N_h+1)]
        us_init = [ddp1.us[i]  for i in range(N_h)]
    else:
        ug = pin_utils.get_u_grav(q0, robot)
        xs_init = [x0 for i in range(N_h+1)]
        us_init = [ug  for i in range(N_h)]
    ddp2.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    print("WITH CRITIC of ITER #"+str(iter_number)+" :  Croco([0,...,T])+V_"+str(iter_number))
    if(bool(WARM_START)):
        print("  ( warm-started from Croco([0,..,"+str(iter_number+1)+"T]) )")
    print("  Cost = ", ddp2.cost)
    print("  V(xT)    = ", ddp2.problem.terminalData.cost)
    print("  V_x(xT)  = ", ddp2.problem.terminalData.Lx)
    print("  V_xx(xT) = \n", ddp2.problem.terminalData.Lxx)
    print("\n")
    # Plot
    if(PLOT):   
        d1 = plot_utils.extract_ddp_data(ddp1)
        d2 = plot_utils.extract_ddp_data(ddp2)
        label1 ='OCP([0,...,'+ str(iter_number+1)+'T])'
        if(bool(WARM_START)):
            label2='OCP([0,...,T]) + V_'+str(iter_number)+' ( warm-started from OCP([0,...,'+str(iter_number+1)+'T]) )'
        else:
            label2='OCP([0,...,T]) + V_'+str(iter_number)
        fig, ax = plot_utils.plot_ddp_results([d1, d2], labels=[label1, label2], SHOW=False, marker='o', sampling_plot=1)
        plot_utils.plot_refs(fig, ax, config, SHOW=True)


if __name__=='__main__':
    # test_trained_single(sys.argv[1], int(sys.argv[2]))
    # test_trained_multiple(sys.argv[1], int(sys.argv[2]), int(sys.argv[-1]))
    check_bellman(sys.argv[1], int(sys.argv[2]), int(sys.argv[-1])) #, sys.argv[4])

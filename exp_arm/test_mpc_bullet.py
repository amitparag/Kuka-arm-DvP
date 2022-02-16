# This files contains the code to test out the learned VF in MPC setting in PyBullet
# Author : Sébastien Kleff
# Date : 09/21/2021

import numpy as np
from utils import path_utils, ocp_utils, plot_utils, pin_utils, mpc_utils
import torch
np.set_printoptions(precision=4, linewidth=180)
import os
from datagen import samples_uniform_IK, samples_uniform
from utils import path_utils
import time
import pybullet as p

# # # # # # # # # # # # # # # # # # #
### LOAD ROBOT MODEL and SIMU ENV ### 
# # # # # # # # # # # # # # # # # # # 
# Read config file
config = path_utils.load_config_file('static_reaching_task_mpc2')
# Load params
dt_simu = 1./float(config['simu_freq'])  
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
N_h = config['N_h']
dt = config['dt'] 
# Simulator and pin wrapper
pybullet_simulator = mpc_utils.init_kuka_simulator(dt=dt_simu, x0=x0)
robot = pybullet_simulator.pin_robot 
nq=robot.model.nq; nv=robot.model.nv; nu=nq; nx=nq+nv
id_ee = robot.model.getFrameId('contact')
mpc_utils.display_target(config['p_des'])
print("-------------------------------------------------------------------")
print("[PyBullet] Created robot (id = "+str(pybullet_simulator.robotId)+")")
print("-------------------------------------------------------------------")
# Load trained NN
resultspath = path_utils.results_path()
path = os.path.join(resultspath, 'trained_models/dvp/Order_1/Horizon_200/eps_9.pth')
Net  = torch.load(path)
print("[DVP] Loaded trained VF.")

# Init OCP 
ddp = ocp_utils.init_DDP(robot, config, x0, critic=Net, 
                                            callbacks=False, 
                                            which_costs=config['WHICH_COSTS'],
                                            dt=dt, N_h=N_h) 
# ug = pin_utils.get_u_grav(x0[:nq], robot)
# ddp.problem.x0 = x0
# ddp.solve( [x0 for i in range(N_h+1)] , [ug  for i in range(N_h)], maxiter=config['maxiter'], isFeasible=False)


# # # # # # # # # # #
### INIT MPC SIMU ###
# # # # # # # # # # #
sim_data = {}
  # Get frequencies
sim_data['plan_freq'] = config['plan_freq']
sim_data['ctrl_freq'] = config['ctrl_freq']
sim_data['simu_freq'] = config['simu_freq']
freq_SIMU = sim_data['simu_freq']
freq_CTRL = sim_data['ctrl_freq']
freq_PLAN = sim_data['plan_freq']
  # Replan & control counters
nb_plan = 0
nb_ctrl = 0
sim_data['T_tot'] = config['T_tot']                     # Total duration of simulation (s)
sim_data['N_plan'] = int(sim_data['T_tot']*freq_PLAN)   # Total number of planning steps in the simulation
sim_data['N_ctrl'] = int(sim_data['T_tot']*freq_CTRL)   # Total number of control steps in the simulation 
sim_data['N_simu'] = int(sim_data['T_tot']*freq_SIMU)   # Total number of simulation steps 
sim_data['T_h'] = config['N_h']*config['dt']            # Duration of the MPC horizon (s)
sim_data['N_h'] = config['N_h']                         # Number of nodes in MPC horizon
sim_data['dt'] = config['dt']
sim_data['dt_ctrl'] = float(1./freq_CTRL)               # Duration of 1 control cycle (s)
sim_data['dt_plan'] = float(1./freq_PLAN)               # Duration of 1 planning cycle (s)
sim_data['dt_simu'] = dt_simu                           # Duration of 1 simulation cycle (s)
# Misc params
sim_data['nq'] = nq
sim_data['nv'] = nv
sim_data['nx'] = nx
sim_data['p_ref'] = config['p_des']
# Main data to record 
sim_data['X_pred'] = np.zeros((sim_data['N_plan'], config['N_h']+1, nx))     # Predicted states (output of DDP, i.e. ddp.xs)
sim_data['U_pred'] = np.zeros((sim_data['N_plan'], config['N_h'], nu))       # Predicted torques (output of DDP, i.e. ddp.us)
sim_data['U_ref'] = np.zeros((sim_data['N_ctrl'], nu))             # Reference torque for motor drivers (i.e. ddp.us[0] interpolated to control frequency)
sim_data['X_mea'] = np.zeros((sim_data['N_simu']+1, nx))           # Measured states (i.e. measured from PyBullet at simu/HF)
sim_data['X_mea'][0, :] = x0

# # # # # # # # # # # #
### SIMULATION LOOP ###
# # # # # # # # # # # #
if(config['INIT_LOG']):
  print('                  ***********************')
  print('                  * Simulation is ready *') 
  print('                  ***********************')        
  print("-------------------------------------------------------------------")
  print('- Total simulation duration            : T_tot  = '+str(sim_data['T_tot'])+' s')
  print('- Simulation frequency                 : f_simu = '+str(float(freq_SIMU/1000.))+' kHz')
  print('- Control frequency                    : f_ctrl = '+str(float(freq_CTRL/1000.))+' kHz')
  print('- Replanning frequency                 : f_plan = '+str(float(freq_PLAN/1000.))+' kHz')
  print('- Total # of simulation steps          : N_ctrl = '+str(sim_data['N_simu']))
  print('- Total # of control steps             : N_ctrl = '+str(sim_data['N_ctrl']))
  print('- Total # of planning steps            : N_plan = '+str(sim_data['N_plan']))
  print('- Duration of MPC horizon              : T_ocp  = '+str(sim_data['T_h'])+' s')
  print('- OCP integration step                 : dt     = '+str(config['dt'])+' s')
  print("-------------------------------------------------------------------")
  print("Simulation will start...")
  time.sleep(config['init_log_display_time'])

# SIMULATE
log_rate = 1000
time_stop_noise = sim_data['T_tot']

for i in range(sim_data['N_simu']): 

    if(i%log_rate==0): 
      print("  ")
      print("SIMU step "+str(i)+"/"+str(sim_data['N_simu']))

  # Solve OCP if we are in a planning cycle (MPC frequency & control frequency)
    if(i%int(freq_SIMU/freq_PLAN) == 0):
        # Reset x0 to measured state + warm-start solution
        ddp.problem.x0 = sim_data['X_mea'][i, :]
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
        xs_init[0] = sim_data['X_mea'][i, :]
        us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
        # Solve OCP & record MPC predictions
        ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
        sim_data['X_pred'][nb_plan, :, :] = np.array(ddp.xs)
        sim_data['U_pred'][nb_plan, :, :] = np.array(ddp.us)
        # Extract desired control torque + prepare interpolation to control frequency
        x_pred_1 = sim_data['X_pred'][nb_plan, 1, :]
        u_plan = sim_data['U_pred'][nb_plan, 0, :]
        # Increment planning counter
        nb_plan += 1
        
  # If we are in a control cycle select reference torque to send to motors
    if(i%int(freq_SIMU/freq_CTRL) == 0):
        u_ctrl = u_plan
        # Record reference torque
        sim_data['U_ref'][nb_ctrl, :] = u_ctrl 
        # Increment control counter
        nb_ctrl += 1
        
  # Simulate actuation with PI torque tracking controller (low-level control frequency)
    u_simu = u_ctrl  
    # Record measured torque & step simulator
    pybullet_simulator.send_joint_command(u_simu)
    p.stepSimulation()
    # Measure new state from simulation 
    q_mea, v_mea = pybullet_simulator.get_state()
    # Update pinocchio model
    pybullet_simulator.forward_robot(q_mea, v_mea)
    # Record data 
    x_mea = np.concatenate([q_mea, v_mea]).T 
    sim_data['X_mea'][i+1, :] = x_mea 

print('--------------------------------')
print('Simulation exited successfully !')
print('--------------------------------')

# # # # # # # # # # # #
# PROCESS SIM RESULTS #
# # # # # # # # # # # #
# Post-process EE trajectories + record in sim data
print('Post-processing end-effector trajectories...')
sim_data['P_EE_pred'] = np.zeros((sim_data['N_plan'], config['N_h']+1, 3))
sim_data['V_EE_pred'] = np.zeros((sim_data['N_plan'], config['N_h']+1, 3))
for node_id in range(config['N_h']+1):
  sim_data['P_EE_pred'][:, node_id, :] = pin_utils.get_p(sim_data['X_pred'][:, node_id, :nq], robot.model, id_ee) - np.array([sim_data['p_ref']]*sim_data['N_plan'])
  sim_data['V_EE_pred'][:, node_id, :] = pin_utils.get_v(sim_data['X_pred'][:, node_id, :nq], sim_data['X_pred'][:, node_id, nv:], robot.model, id_ee)
sim_data['P_EE_mea'] = pin_utils.get_p(sim_data['X_mea'][:,:nq], robot.model, id_ee)
sim_data['V_EE_mea'] = pin_utils.get_v(sim_data['X_mea'][:,:nq], sim_data['X_mea'][:,nv:], robot.model, id_ee)


# # # # # # # # # # #
# PLOT SIM RESULTS  #
# # # # # # # # # # #
save_dir = '/home/skleff/misc_repos/arm/results'
save_name = 'horizon='+str(N_h)+'_3'
# Extract plot data from sim data
plot_data = plot_utils.extract_plot_data(sim_data)
# Plot results
plot_utils.plot_mpc_results(plot_data, which_plots=['x','u','p'],
                              PLOT_PREDICTIONS=True, 
                              pred_plot_sampling=50, #int(plan_freq/500),
                              SAVE=True,
                              SAVE_DIR=save_dir,
                              SAVE_NAME=save_name,
                              AUTOSCALE=True)
# Save optionally
if(config['SAVE_DATA']):
  plot_utils.save_data(sim_data, save_name=save_name, save_dir=save_dir)

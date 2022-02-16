import crocoddyl
import numpy as np
import pinocchio as pin

import utils_amit
from pinocchio.robot_wrapper import RobotWrapper
#from robot_properties_kuka.config import IiwaConfig

import matplotlib.pyplot as plt

np.set_printoptions(precision=4, linewidth=180)

# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
config = utils_amit.load_config_file('static_reaching_task_ocp')
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
# Get pin wrapper
#robot = IiwaConfig.buildRobotWrapper() #  RobotWrapper.BuildFromURDF(...)  # !!! Make your wrapper here !!!
# Get initial frame placement + dimensions of joint space


urdf_path = '/home/amit/DEVEL/baselines/exp_arm/robot_properties_kuka/urdf/iiwa.urdf'
mesh_path = '/home/amit/DEVEL/baselines/exp_arm/robot_properties_kuka' 
#pin_robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_path)
robot   =   RobotWrapper.BuildFromURDF(urdf_path, mesh_path)



id_endeff = robot.model.getFrameId('contact')
nq, nv = robot.model.nq, robot.model.nv
nx = nq+nv
nu = nq
# Update robot model with initial state
robot.framesForwardKinematics(q0)
robot.computeJointJacobians(q0)

# Horizons to be tested
HORIZONS = [50, 60, 70, 80, 90, 100, 300, 500, 800, 1000] #, 1500, 2000] #, 3000, 5000]
#HORIZONS = [50,60,70]
DDPS = []
COSTS = []
for N_h in HORIZONS:
    # Create solver with custom horizon
    ddp = utils_amit.init_DDP(robot, config, x0, callbacks=True, 
                                            which_costs=['translation', 
                                                         'ctrlReg', 
                                                         'stateReg', 
                                                        # 'velocity',
                                                         'stateLim'],
                                            dt = None, N_h=N_h) 
    # Warm-start
    ug = utils_amit.get_u_grav(q0, robot)
    xs_init = [x0 for i in range(N_h+1)]
    us_init = [ug  for i in range(N_h)]
    # Solve
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    
    # Print VF and record
    COSTS.append(ddp.cost)
    DDPS.append(ddp)
    #plot_ddp_results(ddp,robot)

# Plot results
fig, ax = utils_amit.plot_ddp_results(DDPS, robot, which_plots=['x','u','p'], SHOW=True)

# Plot VF
fig, ax = plt.subplots(1, 1)
ax.plot(HORIZONS, COSTS, 'ro', label='V.F.')
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', prop={'size': 16})
plt.show()



import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
from utils import path_utils, ocp_utils, plot_utils, pin_utils
from tqdm import tqdm
import torch
import pinocchio as pin

import matplotlib.pyplot as plt
import time
import os

robot = RobotWrapper.BuildFromURDF(path_utils.kuka_urdf_path(), path_utils.kuka_mesh_path())
config = path_utils.load_config_file('static_reaching_task_ocp2')
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
id_endeff = robot.model.getFrameId('contact')
nq, nv = robot.model.nq, robot.model.nv
nx = nq+nv
nu = nq
# Update robot model with initial state
robot.framesForwardKinematics(q0)
robot.computeJointJacobians(q0)



##################

def tensorize(arrays):
    """
    Convert a list of np arrays to torch tensor
    """
    return [torch.tensor(array,dtype=torch.float64, requires_grad=True) for array in arrays]



# Sampling from conservative range of state space
def samples_uniform(nb_samples:int, eps_q=0.9, eps_v=0.01):
    '''
    Samples initial states x = (q,v) within conservative state range
    '''
    print("Sampling "+str(nb_samples)+" states...")
    samples = []
    q_des, _, _ = pin_utils.IK_position(robot, q0, id_endeff, config['p_des'])
    q_lim = np.array([2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543])
    q_min = np.maximum(q_des - eps_q*np.ones(nq), -q_lim) #0.9*np.array([2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543])
    q_max = np.minimum(q_des + eps_q*np.ones(nq), +q_lim) #0.9*np.array([2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543])
    v_min = np.zeros(nv) - eps_v*np.ones(nv) 
    v_max = np.zeros(nv) + eps_v*np.ones(nv) 
    x_min = np.concatenate([q_min, v_min])   
    x_max = np.concatenate([q_max, v_max])   
    for i in range(nb_samples):
        samples.append( np.random.uniform(low=x_min, high=+x_max, size=(nx,)))
    return np.array(samples)

# Sampling from conservative range of task space
def samples_uniform_IK(nb_samples:int, q0=q0, 
                            p_des=config['p_des'], 
                            v_des=config['v_des'], 
                            id_endeff=id_endeff, 
                            eps_p=0.1, eps_v=0.01):
    '''
    Sample task space (EE pos and vel) and apply IK 
    in order to get corresponding joint space samples
    '''
    # Sample several states 
    N_SAMPLES = nb_samples
    TSK_SPACE_SAMPLES = []
    JNT_SPACE_SAMPLES = []
    # Define bounds in cartesian space to sample (p_EE,v_EE) around (p_des,0)
    p_min = p_des - np.ones(3)*eps_p; p_max = p_des + np.ones(3)*eps_p
    v_min = v_des - np.ones(3)*eps_v; v_max = v_des + np.ones(3)*eps_v
    y_min = np.concatenate([p_min, v_min])
    y_max = np.concatenate([p_max, v_max])
    print("Sampling "+str(N_SAMPLES)+" states...")
    # Generate samples (uniform)
    for i in range(N_SAMPLES):
        # Task space sample
        y_EE = np.random.uniform(low=y_min, high=y_max, size=(6,))
        TSK_SPACE_SAMPLES.append( y_EE )
        # Inverse kinematics
        q, _, _ = pin_utils.IK_position(robot, q0, id_endeff, y_EE[:3],
                                        DISPLAY=False, LOGS=False, DT=1e-1, IT_MAX=1000, EPS=1e-6)
        pin.computeJointJacobians(robot.model, robot.data, q)
        robot.framesForwardKinematics(q)
        J_q = pin.getFrameJacobian(robot.model, robot.data, id_endeff, pin.ReferenceFrame.LOCAL) 
        vq = np.linalg.pinv(J_q)[:,:3].dot(y_EE[3:]) 
        x = np.concatenate([q, vq])
        JNT_SPACE_SAMPLES.append( x )
    return JNT_SPACE_SAMPLES

# Sampling from conservative range of task space
def samples_uniform_IK_adaptive(nb_samples:int, q0=q0, 
                                p_des=config['p_des'], 
                                v_des=config['v_des'], 
                                id_endeff=id_endeff, 
                                eps_p=[0.05, 0.15, 0.25], eps_v=[0.005, 0.01, 0.015]):
    # 100 sampled in each 
    '''
    Sample task space (EE pos and vel) and apply IK 
    in order to get corresponding joint space samples
    use 3 boxes in task space
    '''
    # Sample several states 
    N_SAMPLES = nb_samples
    TSK_SPACE_SAMPLES = []
    JNT_SPACE_SAMPLES = []
    # Define bounds in cartesian space to sample (p_EE,v_EE) around (p_des,0)
    p_min = [p_des - np.ones(3)*eps for eps in eps_p]; p_max = [p_des + np.ones(3)*eps for eps in eps_p]
    v_min = [v_des - np.ones(3)*eps for eps in eps_v]; v_max = [v_des + np.ones(3)*eps for eps in eps_v]
    y_min = [np.concatenate([p_min[i], v_min[i]]) for i in range(3)]
    y_max = [np.concatenate([p_max[i], v_max[i]]) for i in range(3)]
    print("Sampling "+str(N_SAMPLES)+" states...")
    # Generate samples (uniform)
    for box in range(3):
        for i in range(N_SAMPLES//3):
            # Task space sample
            y_EE = np.random.uniform(low=y_min[box], high=y_max[box], size=(6,))
            TSK_SPACE_SAMPLES.append( y_EE )
            # Inverse kinematics
            q, _, _ = pin_utils.IK_position(robot, q0, id_endeff, y_EE[:3],
                                            DISPLAY=False, LOGS=False, DT=1e-1, IT_MAX=1000, EPS=1e-6)
            pin.computeJointJacobians(robot.model, robot.data, q)
            robot.framesForwardKinematics(q)
            J_q = pin.getFrameJacobian(robot.model, robot.data, id_endeff, pin.ReferenceFrame.LOCAL) 
            vq = np.linalg.pinv(J_q)[:,:3].dot(y_EE[3:]) 
            x = np.concatenate([q, vq])
            JNT_SPACE_SAMPLES.append( x )
    return JNT_SPACE_SAMPLES

# Sampling from conservative range of task space + joint space
def samples_uniform_mixed_adaptive(nb_samples:int, q0=q0, 
                                   p_des=config['p_des'], 
                                   v_des=config['v_des'], 
                                   id_endeff=id_endeff, 
                                   eps_p_ee=[0.05, 0.15, 0.25], eps_v_ee=[0.005, 0.01, 0.015],
                                   eps_q=0.9, eps_v=0.01):
    '''
    Sample task space (EE pos and vel) and apply IK 
    in order to get corresponding joint space samples
    use 3 boxes in task space + 1 box in joint space
    '''
    # Sample several states 
    N_SAMPLES = nb_samples
    JNT_SPACE_SAMPLES = samples_uniform_IK_adaptive(3*N_SAMPLES//4, q0, p_des, v_des, id_endeff, eps_p_ee, eps_v_ee)
    JNT_SPACE_SAMPLES.extend( samples_uniform(N_SAMPLES//4, eps_q, eps_v) )
    return JNT_SPACE_SAMPLES


def create_train_data(critic=None,horizon=200,nb_samples=400):
    

    points  =   samples_uniform_mixed_adaptive(nb_samples=nb_samples + 1000)
    np.random.shuffle(points)
    
    x0s     =   []
    v       =   []
    vx      =   []
    DDPS    =   []

    rejected = 0
    for x0 in tqdm(points):

        q0 = x0[:nq]
        # Update robot model with initial state
        robot.framesForwardKinematics(q0)
        robot.computeJointJacobians(q0)


        ddp = ocp_utils.init_DDP(robot,
                                  config,
                                  x0,
                                  critic=critic,
                                  callbacks=False, 
                                  which_costs=config['WHICH_COSTS'],
                                  dt = None,
                                  N_h=horizon) 


        ddp.problem.x0  =   x0   
        ug = pin_utils.get_u_grav(q0, robot)
        xs_init = [x0 for i in range(horizon+1)]
        us_init = [ug  for i in range(horizon)]
        # Solve
        ddp.solve(xs_init, us_init, maxiter=1000, isFeasible=False)
        
        if(ddp.x_reg >= 1e-1 or ddp.u_reg >= 1e-1):
            rejected+=1
            print("Rejected !!!")
            continue

        else:
            gradient    =   list(ddp.Vx)[0]
            value       =   ddp.cost
            point       =   x0
            x0s.append( x0 )
            vx.append( gradient )
            v.append( value )
            print(value)

            # Record ddp_data
            ddp_data = plot_utils.extract_ddp_data(ddp)
            DDPS.append(ddp_data)
            
        if len(v) == nb_samples:
            break

    print(f"Rejected {rejected}")
    x0s     =   np.array( x0s )
    vx      =   np.array( vx )
    v       =   np.array( v ).reshape(-1,1)
    print(f"Dataset shape: {v.shape}")

    # Plot every 1/10 training data + add references
    fig, ax = plot_utils.plot_ddp_results(DDPS, which_plots=['x','u','p'], 
                                                SHOW=False, 
                                                sampling_plot=10)
    plot_utils.plot_refs(fig, ax, config, SHOW=False)
    # Save figures
    savepath = os.path.join(os.path.abspath(__file__ + "/../../"), "results/figures")
    fig['p'].savefig(os.path.join(savepath,'p_'+str(time.time())+'_.png'), dpi=200)
    fig['x'].savefig(os.path.join(savepath,'x_'+str(time.time())+'_.png'), dpi=200)
    fig['u'].savefig(os.path.join(savepath,'u_'+str(time.time())+'_.png'), dpi=200)
    plt.close('all')

    return [x0s, v, vx]

def make_training_dataloader(critic=None,horizon=200,nb_samples=400):
    """
    TO be used specifically in training
    """


    datas       =   create_train_data(critic=critic,horizon=horizon,nb_samples=nb_samples)
    datas       =   tensorize(datas)
    dataset     =   torch.utils.data.TensorDataset(*datas)
    dataloader  =   torch.utils.data.DataLoader(dataset,
                                                batch_size=64,
                                                shuffle=True)
    return dataloader


def create_test_data():

    datas    =  create_train_data(critic=None,horizon=800,nb_samples=50)
    datas    =  tensorize(datas)

    savepath = os.path.join(os.path.abspath(__file__ + "/../../"), "results/test_data")
    torch.save(datas, os.path.join(savepath, "test_data.pth"))

if __name__=='__main__':
    create_test_data()




        


    






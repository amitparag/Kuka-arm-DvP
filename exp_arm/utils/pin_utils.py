# This files contains utilities for rigid-body computations based on  pinocchio
# Author : Sébastien Kleff
# Date : 09/20/2021

import numpy as np
import pinocchio as pin
import time

# Get pin grav torque
def get_u_grav(q, pin_robot):
    '''
    Return gravity torque at q
    '''
    return pin.computeGeneralizedGravity(pin_robot.model, pin_robot.data, q)

# Get EE position
def get_p(q, model, id_endeff):
    '''
    Returns end-effector positions given q trajectory 
        q         : joint positions
        robot     : pinocchio model
        id_endeff : id of EE frame
    '''
    N = np.shape(q)[0]
    p = np.empty((N,3))
    data = model.createData()
    for i in range(N):
        pin.forwardKinematics(model, data, q[i])
        pin.updateFramePlacements(model, data)
        p[i,:] = data.oMf[id_endeff].translation.T
    return p

# Get EE velocity
def get_v(q, dq, model, id_endeff):
    '''
    Returns end-effector velocities given q,dq trajectory 
        q         : joint positions
        dq        : joint velocities
        model     : pinocchio model
        id_endeff : id of EE frame
    '''
    N = np.shape(q)[0]
    v = np.empty((N,3))
    jac = np.zeros((6, model.nv))
    data = model.createData()
    for i in range(N):
        # Get jacobian
        pin.computeJointJacobians(model, data, q[i,:])
        jac = pin.getFrameJacobian(model, data, id_endeff, pin.ReferenceFrame.LOCAL) 
        # Get EE velocity
        v[i,:] = jac.dot(dq[i])[:3]
    return v

# Compute inverse kin
def IK_position(robot, q, frame_id, p_des, LOGS=False, DISPLAY=False, DT=1e-2, IT_MAX=1000, EPS=1e-6, sleep=0.01):
    '''
    Inverse kinematics: returns q, v to reach desired position p
    '''
    errs =[]
    for i in range(IT_MAX):  
        if(i%10 == 0 and LOGS==True):
            print("Step "+str(i)+"/"+str(IT_MAX))
        pin.framesForwardKinematics(robot.model, robot.data, q)  
        oMtool = robot.data.oMf[frame_id]          
        oRtool = oMtool.rotation                  
        tool_Jtool = pin.computeFrameJacobian(robot.model, robot.data, q, frame_id)
        o_Jtool3 = oRtool.dot( tool_Jtool[:3,:] )         # 3D Jac of EE in W frame
        o_TG = oMtool.translation - p_des                 # translation err in W frame 
        vq = -np.linalg.pinv(o_Jtool3).dot(o_TG)          # vel in negative err dir
        q = pin.integrate(robot.model,q, vq * DT)         # take step
        if(DISPLAY):
            robot.display(q)                                   
            time.sleep(sleep)
        errs.append(o_TG)
        if(i%10 == 0 and LOGS==True):
            print(np.linalg.norm(o_TG))
        if np.linalg.norm(o_TG) < EPS:
            break    
    return q, vq, errs

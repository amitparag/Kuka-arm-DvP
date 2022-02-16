from bullet_utils.env import BulletEnvWithGround
from robot_properties_kuka.iiwaWrapper import IiwaRobot
import pybullet as p
import pybullet_data
import numpy as np

# Load KUKA arm in PyBullet environment
def init_kuka_simulator(dt=1e3, x0=None):
    '''
    Loads KUKA LBR iiwa model in PyBullet using the 
    Pinocchio-PyBullet wrapper to simplify interactions
    '''
    # Create PyBullet sim environment + initialize sumulator
    env = BulletEnvWithGround(p.GUI, dt=dt)
    pybullet_simulator = IiwaRobot()
    env.add_robot(pybullet_simulator)
    # Initialize
    if(x0 is None):
        q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
        dq0 = np.zeros(pybullet_simulator.robot.model.nv)
    else:
        q0 = x0[:pybullet_simulator.pin_robot.model.nq]
        dq0 = x0[pybullet_simulator.pin_robot.model.nv:]
    pybullet_simulator.reset_state(q0, dq0)
    pybullet_simulator.forward_robot(q0, dq0)
    return pybullet_simulator

def display_target(p_des):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    target =  p.loadURDF("sphere_small.urdf", basePosition=list(p_des), globalScaling=1, useFixedBase=True)
    # Disable collisons
    p.setCollisionFilterGroupMask(target, -1, 0, 0)

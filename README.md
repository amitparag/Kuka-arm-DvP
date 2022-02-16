# What is it ?
Code for Differential Value Programming (DVP) implementation and example on the KUKA LBR IIWA 14 manipulator. 

# Dependencies
- [PyTorch](https://pytorch.org/) >= v1.8
- [Crocoddyl](https://github.com/loco-3d/crocoddyl) >= v1.8.1
- Python 3.6
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio) >= v2.6.3

# How to use it ?

Install PyTorch, Crocoddyl and Pinocchio first if you don't have them already installed, then 

```
git clone https://github.com/skleff1994/arm.git
``` 
Then `cd exp_arm/` and
- First generate a test set by running `python datagen.py`
- Then launch the training using `python main.py`

The trained NN will be saved in `results/trained_models` and figures of the training sets generated throughout iterations will be saved in `results/figures`. 

The directory `config/robot_properties_kuka` contains URDF and meshes information of the robot, and `config/ocp_params` contains sets of parameters describing the OCP for Crocoddyl. The OCP is setup in `utils/ocp_utils.py`


# Acknowledgements
You can find more information on this algorithm in this paper :

Parag, A., Kleff, S., Saci, L., Mansard, N., & Stasse, O. Value learning from trajectory optimization and Sobolev descent : A step toward reinforcement learning with superlinear convergence properties, _International Conference on Robotics and Automation (ICRA) 2022_ [accepted] 



# This files contains the OCP setup using Crocoddyl
# Author : Sébastien Kleff
# Date : 09/20/2021

import crocoddyl
import numpy as np
import pinocchio as pin
from action_model_critic import ActionModelCritic

# Setup OCP and solver using Crocoddyl
def init_DDP(robot, config, x0,critic=None, callbacks=False, which_costs=['all'], dt=None, N_h=None):
    '''
    Initializes OCP and FDDP solver from config parameters and initial state
      - Running cost: EE placement (Mref) + x_reg (xref) + u_reg (uref)
      - Terminal cost: EE placement (Mref) + EE velocity (0) + x_reg (xref)
      Mref = initial frame placement read in config
      xref = initial state read in config
      uref = initial gravity compensation torque (from xref)
      INPUT: 
          robot       : pinocchio robot wrapper
          config      : dict from YAML config file describing task and MPC params
          x0          : initial state of shooting problem
          callbacks   : display Crocoddyl's DDP solver callbacks
          which_costs : which cost terms in the running & terminal cost?
                          'placement', 'velocity', 'stateReg', 'ctrlReg'
                          'stateLim', 'ctrlLim'
      OUTPUT:
        FDDP solver
    '''

    # OCP parameters
    if(dt is None):
      dt = config['dt']                   # OCP integration step (s)    
    if(N_h is None):
      N_h = config['N_h']                 # Number of knots in the horizon 
    # Model params
    id_endeff = robot.model.getFrameId('contact')
    M_ee = robot.data.oMf[id_endeff]
    nq, nv = robot.model.nq, robot.model.nv
    # Construct cost function terms
    # State and actuation models
    state = crocoddyl.StateMultibody(robot.model)
    actuation = crocoddyl.ActuationModelFull(state)
    # State regularization
    if('all' in which_costs or 'stateReg' in which_costs):
      stateRegWeights = np.asarray(config['stateRegWeights'])
      x_reg_ref = np.concatenate([np.asarray(config['q0']), np.asarray(config['dq0'])]) #np.zeros(nq+nv)     
      xRegCost = crocoddyl.CostModelResidual(state, 
                                            crocoddyl.ActivationModelWeightedQuad(stateRegWeights**2), 
                                            crocoddyl.ResidualModelState(state, x_reg_ref, actuation.nu))
    # Control regularization
    if('all' in which_costs or 'ctrlReg' in which_costs):
      ctrlRegWeights = np.asarray(config['ctrlRegWeights'])
      uRegCost = crocoddyl.CostModelResidual(state, 
                                            crocoddyl.ActivationModelWeightedQuad(ctrlRegWeights**2), 
                                            crocoddyl.ResidualModelControlGrav(state))
    # State limits penalization
    if('all' in which_costs or 'stateLim' in which_costs):
      x_lim_ref  = np.zeros(nq+nv)
      q_max = 0.95*state.ub[:nq] # 95% percent of max q
      v_max = np.ones(nv)        # [-1,+1] for max v
      x_max = np.concatenate([q_max, v_max]) # state.ub
      stateLimWeights = np.asarray(config['stateLimWeights'])
      xLimitCost = crocoddyl.CostModelResidual(state, 
                                            crocoddyl.ActivationModelWeightedQuadraticBarrier(crocoddyl.ActivationBounds(-x_max, x_max),stateLimWeights), 
                                            crocoddyl.ResidualModelState(state, x_lim_ref, actuation.nu))
    # Control limits penalization
    if('all' in which_costs or 'ctrlLim' in which_costs):
      u_min = -np.asarray(config['ctrl_lim']) 
      u_max = +np.asarray(config['ctrl_lim']) 
      u_lim_ref = np.zeros(nq)
      uLimitCost = crocoddyl.CostModelResidual(state, 
                                              crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(u_min, u_max)), 
                                              crocoddyl.ResidualModelControl(state, u_lim_ref))
      # print("[OCP] Added ctrl lim cost.")
    # End-effector placement 
    if('all' in which_costs or 'placement' in which_costs):
      p_target = np.asarray(config['p_des']) 
      desiredFramePlacement = pin.SE3(M_ee.rotation, p_target)
      framePlacementWeights = np.asarray(config['framePlacementWeights'])
      framePlacementCost = crocoddyl.CostModelResidual(state, 
                                                      crocoddyl.ActivationModelWeightedQuad(framePlacementWeights**2), 
                                                      crocoddyl.ResidualModelFramePlacement(state, 
                                                                                            id_endeff, 
                                                                                            desiredFramePlacement, 
                                                                                            actuation.nu)) 
    # End-effector velocity
    if('all' in which_costs or 'velocity' in which_costs): 
      desiredFrameMotion = pin.Motion(np.concatenate([np.asarray(config['v_des']), np.zeros(3)]))
      frameVelocityWeights = np.ones(6)
      frameVelocityCost = crocoddyl.CostModelResidual(state, 
                                                      crocoddyl.ActivationModelWeightedQuad(frameVelocityWeights**2), 
                                                      crocoddyl.ResidualModelFrameVelocity(state, 
                                                                                          id_endeff, 
                                                                                          desiredFrameMotion, 
                                                                                          pin.LOCAL, 
                                                                                          actuation.nu)) 
    # Frame translation cost
    if('all' in which_costs or 'translation' in which_costs):
      desiredFrameTranslation = np.asarray(config['p_des']) 
      frameTranslationWeights = np.asarray(config['frameTranslationWeights'])
      frameTranslationCost = crocoddyl.CostModelResidual(state, 
                                                      crocoddyl.ActivationModelWeightedQuad(frameTranslationWeights**2), 
                                                      crocoddyl.ResidualModelFrameTranslation(state, 
                                                                                            id_endeff, 
                                                                                            desiredFrameTranslation, 
                                                                                            actuation.nu)) 

    # Create IAMs
    runningModels = []
    for i in range(N_h):
        # Create IAM 
        runningModels.append(crocoddyl.IntegratedActionModelEuler( 
            crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                             actuation, 
                                                             crocoddyl.CostModelSum(state, nu=actuation.nu)), dt ) )
        # Add cost models
        if('all' in which_costs or 'placement' in which_costs):
          runningModels[i].differential.costs.addCost("placement", framePlacementCost, config['framePlacementWeight'])
        if('all' in which_costs or 'translation' in which_costs):
          runningModels[i].differential.costs.addCost("translation", frameTranslationCost, config['frameTranslationWeight'])
        if('all' in which_costs or 'velocity' in which_costs):
          runningModels[i].differential.costs.addCost("velocity", frameVelocityCost, config['frameVelocityWeight'])
        if('all' in which_costs or 'stateReg' in which_costs):
          runningModels[i].differential.costs.addCost("stateReg", xRegCost, config['stateRegWeight'])
        if('all' in which_costs or 'ctrlReg' in which_costs):
          runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, config['ctrlRegWeight'])
        if('all' in which_costs or 'stateLim' in which_costs):
          runningModels[i].differential.costs.addCost("stateLim", xLimitCost, config['stateLimWeight'])
        if('all' in which_costs or 'ctrlLim' in which_costs):
          runningModels[i].differential.costs.addCost("ctrlLim", uLimitCost, config['ctrlLimWeight'])
        # Add armature
        runningModels[i].differential.armature = np.asarray(config['armature'])
    
    # Terminal IAM + set armature
    if critic is None:
      terminalModel = crocoddyl.IntegratedActionModelEuler(
          crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                              actuation, 
                                                              crocoddyl.CostModelSum(state, nu=actuation.nu) ) )
      terminalModel.differential.armature = np.asarray(config['armature']) 
    else:
      terminalModel = ActionModelCritic(critic=critic,nx=14)
    
    # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    # Creating the DDP solver 
    ddp = crocoddyl.SolverFDDP(problem)

    if(callbacks):
      ddp.setCallbacks([crocoddyl.CallbackLogger(),
                        crocoddyl.CallbackVerbose()])

    return ddp

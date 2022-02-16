################################################ CRITIC ACTION MODEL   ##########################################
import crocoddyl as C
import numpy as np

class ActionModelCritic(C.ActionModelAbstract):
    """
    Terminal Model with a neural network that predicts cost, gradient and hessian at the terminal position.

    """
    def __init__(self,critic,nx:int,nu:int=1,gauss_approximation=False):
        """
        :param  agent                   torch nn, the value network to use.
        :param  nx                      int, number of input dimensions. dimensions of state space
        :param  nu                      int, dimensions of action space. This is only used to instantiate the
                                        C.ActionModelAbstract.
                                        The params nx and nu are needed by Crocoddyl to make a terminal model
        :param  gauss_approximation     bool, If gauss_approximation, the calculate the jacobian and hessian through
                                        gauss-newton method.

        The default discount_factor for terminal critic layer is 1. When using discount factor on terminal model,
        then the discount_factor have to be changed:= model.discount_factor = discount_factor

        """
        C.ActionModelAbstract.__init__(self, C.StateVector(nx), nu)

        self.critic                  =   critic
        self.approximation           =   gauss_approximation

       
    def calc(self,data,x,u=None):
        data.cost       =  self.critic.calc(x,approximation=self.approximation)
    def calcDiff(self,data,x,u=None):

        data.Lx          =   self.critic.Lx.squeeze()
        data.Lxx         =   self.critic.Lxx.squeeze()
##################################################################################################################
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.autograd.functional as F
import numpy as np
#############################################   CRITIC NETWORK    ###########################################




class Critic(nn.Module):

    def __init__(self,nx:int=3,nr:int=3,nh:int=2,nhu:int=64):
        super(Critic,self).__init__()

        """
        # Creates a coefficient wise Smooth Residual Network.

        :param  nx          :   int, number of input dimensions
        :param  nr          :   int, number of residual outputs
        :param  nh          :   int, number of hidden layers
        :param  nhu         :   int, number of units in hidden layers

        """
        self.nx_                =   nx
        self.nr_                =   nr
        self.residual_network   =   self._architecture(nx=nx,nr=nr,nh=nh,nhu=nhu)

    @property
    def nx(self):
        return self.nx_
    

    @property
    def nr(self):
        return self.nr_


    @staticmethod
    def _architecture(nx,nr,nh,nhu):
        """
        Instantiate a basic neural network
        """
        activation  =   torch.nn.Tanh()

        layers = [
            (f'Input Layer ', nn.Linear(in_features=nx ,out_features=nhu)),
            (f'Input Layer activation', activation),
        ]
        
        # Hidden layers
        for nl in range(nh):
            layers.extend( [
                (f'hidden layer # {nl+1}', nn.Linear(in_features= nhu ,out_features=nhu)),
                (f'hidden layer # {nl+1} activation', activation),
            ])

        layers.append( (f'Residual layer ', nn.Linear(in_features= nhu ,out_features=nr)) )
        
        layers  =   OrderedDict(layers)
        network =   nn.Sequential(layers)

        return network

    @staticmethod
    def pseudonorm(r):
        """
        Return the pseudo norm
        """
        return torch.sqrt((torch.square(r)+1))-1

    @staticmethod
    def dpseudonorm(r,ar):
        """
        return the derivative of pseudonorm of residual and activation(residual)
        
        r   ==  residual
        ar  ==  pseudonorm(residual)

        """
        return (r/(ar+1), 1/(ar+1)**3)

    
    def tensorize(self,x):
        """
        To make sure that x is a tensor
        """
        if not torch.is_tensor(x):
            x   =   torch.tensor(x,requires_grad=True,dtype=torch.float64).reshape(-1,self.nx_)
        if x.ndim == 1:
            x = x.view(-1,self.nx_)

        try:
            assert x.requires_grad == True
        except AssertionError:
            x = x.clone().detach().requires_grad_(True)
        return x

      

    def forward(self,x,order=0):
        """
        If order is 0, then classical regression
        elif order == 1, Sobolev Regression.
        """
        x   =   self.tensorize(x)
        r   =   self.residual_network(x)
        sr  =   self.pseudonorm(r)
        v   =   torch.sum(sr,axis=1).unsqueeze(1)

        if order == 1:
            dv  =   torch.autograd.grad(outputs = v, inputs = x, 
                                        retain_graph=True,grad_outputs=torch.ones_like(v), 
                                        create_graph=True)[0]


            return v,dv
        else:return v


    ### NUMPY ACCESS FOR CRCOODDYL
    def v(self,x):
        r   = self.residual_network(x)
        sr  = self.pseudonorm(r)
        v   = torch.sum(sr,axis=1)
        return v

    def exact_derivatives(self,x):
        """
        Take the hessian and the gradient of the network output with respect to input
        """
        v,dv    =   self.forward(x,order=1)

        d2v     =   F.hessian(self.v, x.view(-1,self.nx_),create_graph=True, \
                              strict=True).squeeze()

        dv      =   dv.detach().numpy().reshape(-1,self.nx_)
        d2v     =   d2v.detach().numpy().reshape(self.nx_,self.nx_)

        return v.item(), dv, d2v

    def inexact_derivatives(self,x):
        """
        Gauss Approximation
        """
        x   =   x.squeeze()
        r   =   self.residual_network(x)
        sr  =   self.pseudonorm(r)
        v   =   torch.sum(sr)

        dr, dsr =   self.dpseudonorm(r,sr)
        r_x     =   torch.autograd.functional.jacobian(self.residual_network,x)

        dv      =   r_x.T @ dr
        
        d2v     =   torch.einsum('k,ki,kj->ij', dsr,r_x,r_x )  

        dv      =   dv.detach().numpy().reshape(-1, self.nx_)
        d2v     =   d2v.detach().numpy().reshape(self.nx_, self.nx_)

        return v.item(), dv, d2v

    def calc(self,x,approximation=False):
        x               =   self.tensorize(x)
        if approximation:
            v,dv,d2v    =   self.inexact_derivatives(x)
        else:
            v,dv,d2v    =   self.exact_derivatives(x)

        self.Lx     =   np.array(dv)
        self.Lxx    =   np.array(d2v).squeeze()
        return v

    def calcDiff(self,x=None):
        if x is not None:
            self.calc(x=x,approximation=True)
        
        return self.Lx,self.Lxx



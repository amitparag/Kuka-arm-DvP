# This files contains useful functions to plot results of OCP
# Author : Sébastien Kleff
# Date : 09/20/2021

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib
import numpy as np
import utils.pin_utils
import time 
import os





# Extract relevant data from DDP solver for plotting
def extract_ddp_data(ddp):
    '''
    Record relevant data from ddp solver in order to plot 
    '''
    # Store data
    ddp_data = {}
    # OCP params
    ddp_data['T'] = ddp.problem.T
    ddp_data['dt'] = ddp.problem.runningModels[0].dt
    ddp_data['nq'] = ddp.problem.runningModels[0].state.nq
    ddp_data['nv'] = ddp.problem.runningModels[0].state.nv
    ddp_data['nu'] = ddp.problem.runningModels[0].differential.actuation.nu
    ddp_data['nx'] = ddp.problem.runningModels[0].state.nx
    # Pin model
    ddp_data['pin_model'] = ddp.problem.runningModels[0].differential.pinocchio
    ddp_data['frame_id'] = ddp.problem.runningModels[0].differential.costs.costs['translation'].cost.residual.id
    # Solution trajectories
    ddp_data['xs'] = ddp.xs
    ddp_data['us'] = ddp.us
    return ddp_data


# Plot results from DDP solver 
def plot_ddp_results(DDPS_DATA, which_plots='all', labels=None, SHOW=False, marker=None, sampling_plot=1):
    '''
    Plot ddp results from 1 or several DDP solvers
    X, U, EE trajs
    INPUT 
      DDPS_DATA    : DDP solver data or list of ddp solvers data
      robot       : pinocchio robot wrapper
      name_endeff : name of end-effector (in pin model) 
    '''
    if(type(DDPS_DATA) != list):
        DDPS_DATA = [DDPS_DATA]
    if(labels==None):
        labels=[None for k in range(len(DDPS_DATA))]
    for k,d in enumerate(DDPS_DATA):
        # Return figs and axes object in case need to overlay new plots
        if(k==0):
            if('x' in which_plots or which_plots =='all'):
                fig_x, ax_x = plot_ddp_state(DDPS_DATA[k], label=labels[k], SHOW=False, marker=marker)
            if('u' in which_plots or which_plots =='all'):
                fig_u, ax_u = plot_ddp_control(DDPS_DATA[k], label=labels[k], SHOW=False, marker=marker)
            if('p' in which_plots or which_plots =='all'):
                fig_p, ax_p = plot_ddp_endeff(DDPS_DATA[k], label=labels[k], SHOW=False, marker=marker)

        # Overlay on top of first plot
        else:
            if(k%sampling_plot==0):
                if('x' in which_plots or which_plots =='all'):
                    plot_ddp_state(DDPS_DATA[k], fig=fig_x, ax=ax_x, label=labels[k], SHOW=False, marker=marker)
                if('u' in which_plots or which_plots =='all'):
                    plot_ddp_control(DDPS_DATA[k], fig=fig_u, ax=ax_u, label=labels[k], SHOW=False, marker=marker)
                if('p' in which_plots or which_plots =='all'):
                    plot_ddp_endeff(DDPS_DATA[k], fig=fig_p, ax=ax_p, label=labels[k], SHOW=False, marker=marker)

    if(SHOW):
      plt.show()
    
    fig = {}
    fig['p'] = fig_p
    fig['x'] = fig_x
    fig['u'] = fig_u

    ax = {}
    ax['p'] = ax_p
    ax['x'] = ax_x
    ax['u'] = ax_u

    return fig, ax
 
def plot_ddp_state(ddp_data, fig=None, ax=None, label=None, SHOW=True, marker=None):
    '''
    Plot ddp_data results (state)
    '''
    # Parameters
    N = ddp_data['T']
    dt = ddp_data['dt'] 
    nq = ddp_data['nq'] 
    nv = ddp_data['nv']
    x = np.array(ddp_data['xs'])
    # Extract pos, vel trajs
    q = x[:,:nq]
    v = x[:,nv:]
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nq, 2, sharex='col', figsize=(19.2,10.8))
    if(label is None):
        label='State'
    for i in range(nq):
        # Positions
        ax[i,0].plot(tspan, q[:,i], linestyle='-', marker=marker, label=label)
        ax[i,0].set_ylabel('$q_%s$'%i, fontsize=16)
        ax[i,0].grid(True)
        # Velocities
        ax[i,1].plot(tspan, v[:,i], linestyle='-', marker=marker, label=label)
        ax[i,1].set_ylabel('$v_%s$'%i, fontsize=16)
        ax[i,1].grid(True)
    ax[-1,0].set_xlabel('Time (s)', fontsize=16)
    ax[-1,1].set_xlabel('Time (s)', fontsize=16)
    fig.align_ylabels(ax[:, 0])
    fig.align_ylabels(ax[:, 1])
    # Legend
    handles, labels = ax[i,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('State : joint positions and velocities', size=18)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_control(ddp_data, fig=None, ax=None, label=None, SHOW=True, marker=None):
    '''
    Plot ddp_data results (control)
    '''
    N = ddp_data['T'] 
    dt = ddp_data['dt'] 
    nu = ddp_data['nu'] 
    u = np.array(ddp_data['us'])
    # Plots
    tspan = np.linspace(0, N*dt-dt, N)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nu, 1, sharex='col', figsize=(19.2,10.8))
    if(label is None):
        label='Control'    
    for i in range(nu):
        # Positions
        ax[i].plot(tspan, u[:,i], linestyle='-', marker=marker, label=label)
        ax[i].set_ylabel('$u_%s$'%i, fontsize=16)
        ax[i].grid(True)
    ax[-1].set_xlabel('Time (s)', fontsize=16)
    fig.align_ylabels(ax[:])
    handles, labels = ax[i].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('Control trajectories', size=18)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_endeff(ddp_data, fig=None, ax=None, label=None, SHOW=True, marker=None):
    '''
    Plot ddp_data results (endeff)
    '''
    # Parameters
    N = ddp_data['T'] 
    dt = ddp_data['dt'] 
    nq = ddp_data['nq'] 
    x = np.array(ddp_data['xs'])
    # Extract EE traj
    q = x[:,:nq]
    v = x[:,nq:]
    p_EE = utils.pin_utils.get_p(q, ddp_data['pin_model'], ddp_data['frame_id'])
    v_EE = utils.pin_utils.get_v(q, v, ddp_data['pin_model'], ddp_data['frame_id'])
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(3, 2, sharex='col', figsize=(19.2,10.8))
    if(label is None):
        label='End-effector'
    xyz = ['x','y','z']
    for i in range(3):
        # Positions
        ax[i,0].plot(tspan, p_EE[:,i], linestyle='-', marker=marker)#, label=label)
        ax[i,0].set_ylabel('$P^{EE}_%s$ (m)'%xyz[i], fontsize=16)
        ax[i,0].grid(True)
        #Velocities
        ax[i,1].plot(tspan, v_EE[:,i], linestyle='-', marker=marker)#, label=label)
        ax[i,1].set_ylabel('$V^{EE}_%s$ (m)'%xyz[i], fontsize=16)
        ax[i,1].grid(True)
    ax[-1,0].set_xlabel('Time (s)', fontsize=16)
    ax[-1,1].set_xlabel('Time (s)', fontsize=16)
    fig.align_ylabels(ax[:,0])
    fig.align_ylabels(ax[:,1])
    handles, labels = ax[i,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('End-effector positions and velocities', size=18)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_refs(fig, ax, config, SHOW=True):
    '''
    Overlay references on top of existing plots
    '''

    dt = config['dt']; N_h = config['N_h']
    nq = len(config['q0']); nu = nq
    # Add EE refs
    xyz = ['x','y','z']
    for i in range(3):
        ax['p'][i,0].plot(np.linspace(0, N_h*dt, N_h+1), [np.asarray(config['p_des']) [i]]*(N_h+1), 'r-.', label='Desired')
        ax['p'][i,0].set_ylabel('$P^{EE}_%s$ (m)'%xyz[i], fontsize=16)
        ax['p'][i,1].plot(np.linspace(0, N_h*dt, N_h+1), [np.asarray(config['v_des']) [i]]*(N_h+1), 'r-.', label='Desired')
        ax['p'][i,1].set_ylabel('$V^{EE}_%s$ (m)'%xyz[i], fontsize=16)
    handles_x, labels_x = ax['p'][i,0].get_legend_handles_labels()
    fig['p'].legend(handles_x, labels_x, loc='upper right', prop={'size': 16})

    # Add vel refs
    for i in range(nq):
        # ax['x'][i,0].plot(np.linspace(0*dt, N_h*dt, N_h+1), [np.asarray(config['q0'])[i]]*(N_h+1), 'r-.', label='Desired')
        ax['x'][i,1].plot(np.linspace(0*dt, N_h*dt, N_h+1), [np.asarray(config['dq0'])[i]]*(N_h+1), 'r-.', label='Desired')

    if(SHOW):
        plt.show()
    
    return fig, ax
    
    # # Add torque refs
    # q = np.array(ddp_data['xs'])[:,:nq]
    # ureg_ref = np.zeros((N_h, nu))
    # for i in range(N_h):
    #     ureg_ref[i,:] = utils.pin_utils.get_u_grav_(q[i,:], ddp_data['pin_model'])
    # for i in range(nu):
    #     ax['u'][i].plot(np.linspace(0*dt, N_h*dt, N_h), ureg_ref[:,i], 'r-.', label='Desired')






# Save data (dict) into compressed npz
def save_data(sim_data, save_name=None, save_dir=None):
    '''
    Saves data to a compressed npz file (binary)
    '''
    print('Compressing & saving data...')
    if(save_name is None):
        save_name = 'sim_data_NO_NAME'+str(time.time())
    if(save_dir is None):
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../data'))
    save_path = save_dir+'/'+save_name+'.npz'
    np.savez_compressed(save_path, data=sim_data)
    print("Saved data to "+str(save_path)+" !")

# Loads dict from compressed npz
def load_data(npz_file):
    '''
    Loads a npz archive of sim_data into a dict
    '''
    print('Loading data...')
    d = np.load(npz_file, allow_pickle=True)
    return d['data'][()]







# Extract MPC simu-specific plotting data from sim data
def extract_plot_data_from_sim_data(sim_data):
    '''
    Extract plot data from simu data
    '''
    print('Extracting plotting data from simulation data...')
    plot_data = {}
    nx = sim_data['X_mea'].shape[1]
    nq = nx//2
    nv = nx-nq
    # Misc. params
    plot_data['T_tot'] = sim_data['T_tot']
    plot_data['N_simu'] = sim_data['N_simu']
    plot_data['N_ctrl'] = sim_data['N_ctrl']
    plot_data['N_plan'] = sim_data['N_plan']
    plot_data['dt_plan'] = sim_data['dt_plan']
    plot_data['dt_ctrl'] = sim_data['dt_ctrl']
    plot_data['dt_simu'] = sim_data['dt_simu']
    plot_data['nq'] = sim_data['nq']
    plot_data['nv'] = sim_data['nv']
    plot_data['nx'] = sim_data['nx']
    plot_data['T_h'] = sim_data['T_h']
    plot_data['N_h'] = sim_data['N_h']
    plot_data['p_ref'] = sim_data['p_ref']
    # state predictions
    plot_data['q_pred'] = sim_data['X_pred'][:,:,:nq]
    plot_data['v_pred'] = sim_data['X_pred'][:,:,nv:]
    # measured state
    plot_data['q_mea'] = sim_data['X_mea'][:,:nq]
    plot_data['v_mea'] = sim_data['X_mea'][:,nv:]
    # end-eff position
    plot_data['p_ee_mea'] = sim_data['P_EE_mea']
    plot_data['p_ee_pred'] = sim_data['P_EE_pred']
    plot_data['v_ee_mea'] = sim_data['V_EE_mea']
    plot_data['v_ee_pred'] = sim_data['V_EE_pred']
    # plot_data['p_des'] = sim_data['P_des'] 
    # control
    plot_data['u_pred'] = sim_data['U_pred']
    plot_data['u_des'] = sim_data['U_pred'][:,0,:]

    return plot_data

# Same giving npz path OR dict as argument
def extract_plot_data(input_data):
    '''
    Extract plot data from npz archive or sim_data
    '''
    if(type(input_data)==str):
        sim_data = load_data(input_data)
    elif(type(input_data)==dict):
        sim_data = input_data
    else:
        TypeError("Input data must be a Python dict or a path to .npz archive")
    return extract_plot_data_from_sim_data(sim_data)



### Plot from MPC simulation 
def plot_mpc_state(plot_data, PLOT_PREDICTIONS=False, 
                          pred_plot_sampling=100, 
                          SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                          SHOW=True):
    '''
    Plot state data
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
    '''
    print('Plotting state data...')
    T_tot = plot_data['T_tot']
    N_simu = plot_data['N_simu']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    nq = plot_data['nq']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu_x = np.linspace(0, T_tot, N_simu+1)
    t_span_plan_x = np.linspace(0, T_tot, N_plan+1)
    fig_x, ax_x = plt.subplots(nq, 2, figsize=(19.2,10.8), sharex='col') 
    # For each joint
    for i in range(nq):

        if(PLOT_PREDICTIONS):

            # Extract state predictions of i^th joint
            q_pred_i = plot_data['q_pred'][:,:,i]
            v_pred_i = plot_data['v_pred'][:,:,i]

            # For each planning step in the trajectory
            for j in range(0, N_plan, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_plan
                tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                tspan_u_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
                # Set up lists of (x,y) points for predicted positions and velocities
                points_q = np.array([tspan_x_pred, q_pred_i[j,:]]).transpose().reshape(-1,1,2)
                points_v = np.array([tspan_x_pred, v_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_q = np.concatenate([points_q[:-1], points_q[1:]], axis=1)
                segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_q = LineCollection(segs_q, cmap=cm, zorder=-1)
                lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
                lc_q.set_array(tspan_x_pred)
                lc_v.set_array(tspan_x_pred) 
                # Customize
                lc_q.set_linestyle('-')
                lc_v.set_linestyle('-')
                lc_q.set_linewidth(1)
                lc_v.set_linewidth(1)
                # Plot collections
                ax_x[i,0].add_collection(lc_q)
                ax_x[i,1].add_collection(lc_v)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax_x[i,0].scatter(tspan_x_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
                ax_x[i,1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',

        # Joint position
        # ax_x[i,0].plot(t_span_plan_x, plot_data['q_des'][:,i], 'b-',  marker='o', label='Desired')
        ax_x[i,0].plot(t_span_simu_x, plot_data['q_mea'][:,i], 'r-', label='Measured ', linewidth=1.5, alpha=0.5)
        ax_x[i,0].set_ylabel('$q_{}$'.format(i), fontsize=12)
        ax_x[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_x[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax_x[i,0].grid(True)
        
        # Joint velocity 
        # ax_x[i,1].plot(t_span_plan_x, plot_data['v_des'][:,i], 'b-',  marker='o', label='Desired')
        ax_x[i,1].plot(t_span_simu_x, plot_data['v_mea'][:,i], 'r-', label='Measured ', linewidth=1.5, alpha=0.5)
        ax_x[i,1].set_ylabel('$v_{}$'.format(i), fontsize=12)
        ax_x[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_x[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax_x[i,1].grid(True)
        # Add xlabel on bottom plot of each column
        if(i == nq-1):
            ax_x[i,0].set_xlabel('t(s)', fontsize=16)
            ax_x[i,1].set_xlabel('t(s)', fontsize=16)
        # Legend
        handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
        fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
    # y axis labels
    fig_x.text(0.05, 0.5, 'Joint position (rad)', va='center', rotation='vertical', fontsize=16)
    fig_x.text(0.49, 0.5, 'Joint velocity (rad/s)', va='center', rotation='vertical', fontsize=16)
    fig_x.subplots_adjust(wspace=0.27)
    # Titles
    fig_x.suptitle('State = joint positions, velocities', size=16)
    # Save fig
    if(SAVE):
        figs = {'x': fig_x}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/misc_repos/arm/results'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig_x

# Plot control data
def plot_mpc_control(plot_data, PLOT_PREDICTIONS=False, 
                            pred_plot_sampling=100, 
                            SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                            SHOW=True,
                            AUTOSCALE=False):
    '''
    Plot control data
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
    '''
    print('Plotting control data...')
    T_tot = plot_data['T_tot']
    N_simu = plot_data['N_simu']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    dt_simu = plot_data['dt_simu']
    nq = plot_data['nq']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu_u = np.linspace(0, T_tot-dt_simu, N_simu)
    t_span_plan_u = np.linspace(0, T_tot-dt_plan, N_plan)
    fig_u, ax_u = plt.subplots(nq, 1, figsize=(19.2,10.8), sharex='col') 
    # For each joint
    for i in range(nq):

        if(PLOT_PREDICTIONS):

            # Extract state predictions of i^th joint
            u_pred_i = plot_data['u_pred'][:,:,i]

            # For each planning step in the trajectory
            for j in range(0, N_plan, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_plan
                tspan_u_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
                # Set up lists of (x,y) points for predicted positions and velocities
                points_u = np.array([tspan_u_pred, u_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_u = np.concatenate([points_u[:-1], points_u[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_u = LineCollection(segs_u, cmap=cm, zorder=-1)
                lc_u.set_array(tspan_u_pred)
                # Customize
                lc_u.set_linestyle('-')
                lc_u.set_linewidth(1)
                # Plot collections
                ax_u[i].add_collection(lc_u)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax_u[i].scatter(tspan_u_pred, u_pred_i[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 

        # Joint torques
        ax_u[i].plot(t_span_plan_u, plot_data['u_des'][:,i], 'b-', label='Desired')
        ax_u[i].set_ylabel('$u_{}$'.format(i), fontsize=12)
        ax_u[i].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_u[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_u[i].grid(True)
        # Last x axis label
        if(i == nq-1):
            ax_u[i].set_xlabel('t (s)', fontsize=16)
        # LEgend
        handles_u, labels_u = ax_u[i].get_legend_handles_labels()
        fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})
    # Sup-y label
    fig_u.text(0.04, 0.5, 'Joint torque (Nm)', va='center', rotation='vertical', fontsize=16)
    # Titles
    fig_u.suptitle('Control = joint torques', size=16)
    # Save figs
    if(SAVE):
        figs = {'u': fig_u}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/misc_repos/arm/results'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 

    return fig_u

# Plot end-eff data
def plot_mpc_endeff(plot_data, PLOT_PREDICTIONS=False, 
                           pred_plot_sampling=100, 
                           SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                           SHOW=True,
                           AUTOSCALE=False):
    '''
    Plot endeff data
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
      AUTOSCALE                 : rescale y-axis of endeff plot 
                                  based on maximum value taken
    '''
    print('Plotting end-eff data...')
    T_tot = plot_data['T_tot']
    N_simu = plot_data['N_simu']
    N_ctrl = plot_data['N_ctrl']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    p_ref = plot_data['p_ref']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu_x = np.linspace(0, T_tot, N_simu+1)
    t_span_ctrl_x = np.linspace(0, T_tot, N_ctrl+1)
    t_span_plan_x = np.linspace(0, T_tot, N_plan+1)
    fig_p, ax_p = plt.subplots(3,2, figsize=(19.2,10.8), sharex='col') 
    if(PLOT_PREDICTIONS):
        # For each component (x,y,z)
        for i in range(3):
            p_pred_i = plot_data['p_ee_pred'][:, :, i]
            v_pred_i = plot_data['v_ee_pred'][:, :, i]
            # For each planning step in the trajectory
            for j in range(0, N_plan, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_plan
                tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                # Set up lists of (x,y) points for predicted positions
                points_p = np.array([tspan_x_pred, p_pred_i[j,:]]).transpose().reshape(-1,1,2)
                points_v = np.array([tspan_x_pred, v_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_p = np.concatenate([points_p[:-1], points_p[1:]], axis=1)
                segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_p = LineCollection(segs_p, cmap=cm, zorder=-1)
                lc_p.set_array(tspan_x_pred)
                lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
                lc_v.set_array(tspan_x_pred)
                # Customize
                lc_p.set_linestyle('-')
                lc_p.set_linewidth(1)
                lc_v.set_linestyle('-')
                lc_v.set_linewidth(1)
                # Plot collections
                ax_p[i,0].add_collection(lc_p)
                ax_p[i,1].add_collection(lc_v)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax_p[i,0].scatter(tspan_x_pred, p_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)
                ax_p[i,1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)
    # Plot endeff
    # x
    # v_EE_des = utils.pin_utils.get_v(plot_data['p_des'], plot_data['p_des'], ddp_data['pin_model'], ddp_data['frame_id'])
    # ax_p[0].plot(t_span_plan_x, plot_data['p_des'][:,0]-p_ref[0], 'b-', label='p_des - p_ref', alpha=0.5)
    xyz = ['x', 'y', 'z']
    for i in range(3):
        ax_p[i,0].plot(t_span_simu_x, plot_data['p_ee_mea'][:,i]-[p_ref[i]]*(N_simu+1), 'r-', label='p_ee_mea - p_ref ', linewidth=1.5, alpha=0.5)
        # ax_p[i,0].set_title('x-position-ERROR')
        ax_p[i,0].set_ylabel('$P^{EE}_%s$ (m)'%xyz[i], fontsize=16)
        ax_p[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_p[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_p[i,0].grid(True)
        # Add frame ref if any
        ax_p[i,0].plot(t_span_ctrl_x, [0.]*(N_ctrl+1), 'g-.', linewidth=2., alpha=0.7)
        ax_p[i,0].plot(t_span_ctrl_x, [0.]*(N_ctrl+1), 'g-.', linewidth=2., alpha=0.7)
        ax_p[i,0].plot(t_span_ctrl_x, [0.]*(N_ctrl+1), 'g-.', linewidth=2., alpha=0.7)

        ax_p[i,1].plot(t_span_simu_x, plot_data['v_ee_mea'][:,i], 'r-', label='v_ee_mea', linewidth=1.5, alpha=0.5)
        # ax_p[i,1].set_title('x-position-ERROR')
        ax_p[i,1].set_ylabel('$V^{EE}_%s$ (m)'%xyz[i], fontsize=16)
        ax_p[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_p[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_p[i,1].grid(True)
        # Add frame ref if any
        ax_p[i,1].plot(t_span_ctrl_x, [0.]*(N_ctrl+1), 'g-.', linewidth=2., alpha=0.7)
        ax_p[i,1].plot(t_span_ctrl_x, [0.]*(N_ctrl+1), 'g-.', linewidth=2., alpha=0.7)
        ax_p[i,1].plot(t_span_ctrl_x, [0.]*(N_ctrl+1), 'g-.', linewidth=2., alpha=0.7)

        ax_p[-1,0].set_xlabel('Time (s)', fontsize=16)
        ax_p[-1,1].set_xlabel('Time (s)', fontsize=16)
        fig_p.align_ylabels(ax_p[:,0])
        fig_p.align_ylabels(ax_p[:,1])

    # Set ylim if any
    # if(AUTOSCALE):
    #     ax_p_ylim = np.max(np.abs(plot_data['p_ee_mea']-plot_data['p_ref']))
    #     ax_p[0].set_ylim(-ax_p_ylim, ax_p_ylim) 
    #     ax_p[1].set_ylim(-ax_p_ylim, ax_p_ylim) 
    #     ax_p[2].set_ylim(-ax_p_ylim, ax_p_ylim) 

    handles_p, labels_p = ax_p[0,0].get_legend_handles_labels()
    fig_p.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})

    # Titles
    fig_p.suptitle('End-effector trajectories errors', size=16)

    # Save figs
    if(SAVE):
        figs = {'p': fig_p}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/misc_repos/arm/results'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig_p

# Plot data
def plot_mpc_results(plot_data, which_plots=None, PLOT_PREDICTIONS=False, 
                                              pred_plot_sampling=100, 
                                              SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                                              SHOW=True,
                                              AUTOSCALE=False):
    '''
    Plot sim data
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
      AUTOSCALE                 : rescale y-axis of endeff plot 
                                  based on maximum value taken
    '''

    plots = {}

    if('x' in which_plots or which_plots is None or which_plots =='all'):
        plots['x'] = plot_mpc_state(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                           pred_plot_sampling=pred_plot_sampling, 
                                           SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                           SHOW=False)
    
    if('u' in which_plots or which_plots is None or which_plots =='all'):
        plots['u'] = plot_mpc_control(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                             pred_plot_sampling=pred_plot_sampling, 
                                             SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                             SHOW=False)

    if('p' in which_plots or which_plots is None or which_plots =='all'):
        plots['p'] = plot_mpc_endeff(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                            pred_plot_sampling=pred_plot_sampling, 
                                            SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False, AUTOSCALE=AUTOSCALE)
    if(SHOW):
        plt.show() 
    plt.close('all')





# Add limits?

# Full script to generate complete plots (handy to have somewhere)
# # State 
# x1 = np.array(ddp1.xs); x2 = np.array(ddp2.xs)
# u1 = np.array(ddp1.us); u2 = np.array(ddp2.us)
# q1 = x1[:,:nq]; v1 = x1[:,nv:]
# q2 = x2[:,:nq]; v2 = x2[:,nv:] 
# fig_x, ax_x = plt.subplots(nq, 2, sharex='col') 
# fig_u, ax_u = plt.subplots(nu, 1, sharex='col') 
# if(bool(WARM_START)):
#     label='Croco(0..T) + V_'+str(iter_number)+' ( warm-started from Croco([0,..,'+str(iter_number+1)+'T]) )'
# else:
#     label='Croco(0..T) + V_'+str(iter_number)
# for i in range(nq):
#     ax_x[i,0].plot(np.linspace(0*dt, (iter_number+1)*N_h*dt, (iter_number+1)*N_h+1), q1[:,i], linestyle='-', marker='o', color='b', label='Croco', alpha=0.5)
#     ax_x[i,0].plot(np.linspace(0*dt, N_h*dt, N_h+1), q2[:,i], linestyle='-', marker='o', color='r', label=label, alpha=0.5)
#     ax_x[i,0].grid(True)
#     ax_x[i,0].set_ylabel('$q_%s$'%i, fontsize=16)
#     ax_x[i,1].plot(np.linspace(0*dt, (iter_number+1)*N_h*dt, (iter_number+1)*N_h+1), v1[:,i], linestyle='-', marker='o', color='b', label='Croco', alpha=0.5)
#     ax_x[i,1].plot(np.linspace(0*dt, N_h*dt, N_h+1), v2[:,i], linestyle='-', marker='o', color='r', label=label, alpha=0.5)
#     ax_x[i,1].grid(True)
#     ax_x[i,1].set_ylabel('$v_%s$'%i, fontsize=16)
#     ax_u[i].plot(np.linspace(0*dt, (iter_number+1)*N_h*dt, (iter_number+1)*N_h), u1[:,i], linestyle='-', marker='o', color='b', label='Croco', alpha=0.5)
#     ax_u[i].plot(np.linspace(0*dt, N_h*dt, N_h), u2[:,i], linestyle='-', marker='o', color='r', label=label, alpha=0.5)
#     ax_u[i].grid(True)
#     ax_u[i].set_ylabel('$u_%s$'%i, fontsize=16)
# ax_x[-1,0].set_xlabel('Time (s)', fontsize=16)
# ax_x[-1,1].set_xlabel('Time (s)', fontsize=16)
# fig_x.align_ylabels(ax_x[:, 0])
# fig_x.align_ylabels(ax_x[:, 1])
# ax_u[-1].set_xlabel('Time (s)', fontsize=16)
# fig_u.align_ylabels(ax_u[:])
# handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
# fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
# fig_x.suptitle('State : joint positions and velocities', fontsize=18)
# handles_u, labels_u = ax_u[0].get_legend_handles_labels()
# fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})
# fig_u.suptitle('Control : joint torques', fontsize=18)

# # EE trajs
# p_ee1 = utils.pin_utils.get_p(q1, robot.model, id_ee)
# p_ee2 = utils.pin_utils.get_p(q2, robot.model, id_ee)
# v_ee1 = utils.pin_utils.get_v(q1, v1, robot.model, id_ee)
# v_ee2 = utils.pin_utils.get_v(q2, v2, robot.model, id_ee)
# fig_p, ax_p = plt.subplots(3, 2, sharex='col') 
# if(bool(WARM_START)):
#     label='Croco(0..T) + V_'+str(iter_number)+' ( warm-started from Croco([0,..,'+str(iter_number+1)+'T]) )'
# else:
#     label='Croco(0..T) + V_'+str(iter_number)
# xyz = ['x','y','z']
# for i in range(3):
#     ax_p[i,0].plot(np.linspace(0*dt, (iter_number+1)*N_h*dt, (iter_number+1)*N_h+1), p_ee1[:,i], linestyle='-', marker='o', color='b', label='Croco', alpha=0.5)
#     ax_p[i,0].plot(np.linspace(0*dt, N_h*dt, N_h+1), p_ee2[:,i], linestyle='-', marker='o', color='r', label=label, alpha=0.5)
#     ax_p[i,0].grid(True)
#     ax_p[i,0].set_ylabel('$P^{EE}_%s$ (m)'%xyz[i], fontsize=16)
#     ax_p[i,1].plot(np.linspace(0*dt, (iter_number+1)*N_h*dt, (iter_number+1)*N_h+1), v_ee1[:,i], linestyle='-', marker='o', color='b', label='Croco', alpha=0.5)
#     ax_p[i,1].plot(np.linspace(0*dt, N_h*dt, N_h+1), v_ee2[:,i], linestyle='-', marker='o', color='r', label=label, alpha=0.5)
#     ax_p[i,1].grid(True)
#     ax_p[i,1].set_ylabel('$V^{EE}_%s$ (m/s)'%xyz[i], fontsize=16)
# ax_p[-1,0].set_xlabel('Time (s)', fontsize=16)
# ax_p[-1,1].set_xlabel('Time (s)', fontsize=16)
# fig_p.align_ylabels(ax_p[:,0])
# fig_p.align_ylabels(ax_p[:,1])
# handles_p, labels_p = ax_p[i,0].get_legend_handles_labels()
# fig_p.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
# fig_p.suptitle('End-effector positions and velocities', fontsize=18)
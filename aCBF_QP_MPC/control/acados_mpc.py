''' Installing acados:
        https://docs.acados.org/installation/index.html#windows-10-wsl
    Installing python interface:
        https://docs.acados.org/python_interface/index.html
    May need to install qpOASES version 3.1 as well.
'''

from acados_template import AcadosOcpSolver, AcadosOcp
import numpy as np
from numpy import pi
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import time

import atexit
import shutil
import os


class AcadosMpc():
    ''' SQP approximation of nonlinear MPC using Acados's OCP solver.
    '''

    def __init__(self, model, Q, R, u_max: np.ndarray, u_min: np.ndarray, time_step: float, num_nodes: int,):
        ''' Initialize the MPC with dynamics as casadi namespace,
            Q & R cost matrices, time-step,
            number of shooting nodes (length of prediction horizon),
            square of maximum motor frequency. 
        '''
        self.nx, self.nu = self.get_model_dims(model)

        assert Q.shape == (self.nx, self.nx)
        assert R.shape == (self.nu, self.nu)
        assert u_max.shape[0] == self.nu
        assert u_min.shape[0] == self.nu
        assert type(num_nodes) == int

        self.DT = time_step
        self.N = num_nodes
        self.u_min = u_min
        self.solver = self.init_solver(model, Q, R, u_max, u_min)

        # deleting acados compiled files when script is terminated.
        atexit.register(self.delete_compiled_files)


    def get_model_dims(self, model):
        ''' Acados model format:
            f_imp_expr/f_expl_expr, x, xdot, u, name 
        '''
        nx = model.x.shape[0]
        nu = model.u.shape[0]
        return nx, nu


    def init_solver(self, model, Q, R, u_max, u_min):
        ''' Guide to acados OCP formulation: 
            https://github.com/acados/acados/blob/master/docs/problem_formulation/problem_formulation_ocp_mex.pdf 
        '''
        nx = self.nx
        nu = self.nu
        ny = nx + nu    # combine x and u into y

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        ocp.dims.nu = nu
        ocp.dims.nx = nx
        ocp.dims.ny = ny
        ocp.dims.nbx_0 = nx
        ocp.dims.nbu = nu
        #ocp.dims.nbx = 4    # number of states being constrained


        # total horizon in seconds
        ocp.solver_options.tf = self.DT*self.N  

        # formulate the default least-squares cost as a quadratic cost
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        # W is a block diag matrix of Q and R costs from standard QP
        ocp.cost.W = np.block([
            [Q, np.zeros((self.nx,self.nu))],
            [np.zeros((self.nu,self.nx)), R],
        ])

        # use V coeffs to map x & u to y
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx,:nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:,-nu:] = np.eye(nu)

        # Initialize reference trajectory (will be overwritten)
        ocp.cost.yref = np.zeros(ny)

        # Initial state (will be overwritten)
        ocp.constraints.x0 = np.zeros(nx)

        # control input constraints (square of motor freq)
        ocp.constraints.lbu = u_min
        ocp.constraints.ubu = u_max
        ocp.constraints.idxbu = np.arange(nu)
        '''
        # state constraints: z, roll, pitch, yaw
        inf = 1000000000
        ocp.constraints.lbx = np.array([0, -pi/2, -pi/2, 0])
        ocp.constraints.ubx = np.array([inf, pi/2, pi/2, 2*pi])
        ocp.constraints.idxbx = np.array([2, 3, 4, 5])
        '''
        # not sure what this is, but this paper say partial condensing HPIPM 
        # is fastest: https://cdn.syscop.de/publications/Frison2020a.pdf
        ocp.solver_options.hpipm_mode = 'SPEED_ABS'
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.qp_solver_iter_max = 50
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.print_level = 0

        # compile acados ocp
        solver = AcadosOcpSolver(ocp)
        return solver


    def run_optimization(self, x0, x_set, timer) -> np.ndarray:
        ''' Set initial state and setpoint,
            then solve the optimization once. 
        '''
        if timer: st = time.perf_counter()

        assert len(x0) == self.nx
        assert len(x_set) == self.nx

        # bound x0 to initial state
        self.solver.set(0, 'lbx', x0)
        self.solver.set(0, 'ubx', x0)
        
        # the reference input will be the hover input
        y_ref = np.concatenate((x_set, self.u_min))
        for k in range(self.N): 
            self.solver.set(k, 'yref', y_ref)
        
        # solve for the next ctrl input
        self.solver.solve()
        if timer: print(f"mpc runtime: {time.perf_counter() - st}")
        return


    def get_next_control(self, x0, x_set, timer=False):
        ''' Get the first control action from the optimization.
        '''
        self.run_optimization(x0, x_set, timer)
        nxt_ctrl = self.solver.get(0, 'u')
        return nxt_ctrl


    def get_next_state(self, x0, x_set, timer=False, visuals=False):
        ''' Get the next state from the optimization.
        '''
        self.run_optimization(x0, x_set, timer)
        nxt_state = self.solver.get(1, 'x')

        if visuals:
            opt_us = np.zeros((self.N, self.nu))
            opt_xs = np.zeros((self.N, self.nx))
            for k in range(self.N):
                opt_us[k] = self.solver.get(k, 'u')
                opt_xs[k] = self.solver.get(k, 'x')
            self.vis_plots(opt_us, opt_xs)
        return nxt_state


    def vis_plots(self, ctrl_inp:np.ndarray, traj:np.ndarray):
        ''' Displaying the series of control inp 
            and trajectory over prediction horizon. 
        '''
        interp_N = 1000
        
        t = self.DT * np.arange(self.N)
        t_interp = self.get_interpolated_curve(
            t, t, interp_N)

        u1 = self.get_interpolated_curve(
            t, ctrl_inp[:,0], interp_N)
        u2 = self.get_interpolated_curve(
            t, ctrl_inp[:,1], interp_N)
        u3 = self.get_interpolated_curve(
            t, ctrl_inp[:,2], interp_N)
        u4 = self.get_interpolated_curve(
            t, ctrl_inp[:,3], interp_N)

        x = self.get_interpolated_curve(
            t, traj[:,0], interp_N)
        y = self.get_interpolated_curve(
            t, traj[:,1], interp_N)
        z = self.get_interpolated_curve(
            t, traj[:,2], interp_N)

        phi = self.get_interpolated_curve(
            t, traj[:,3], interp_N)
        theta = self.get_interpolated_curve(
            t, traj[:,4], interp_N)
        psi = self.get_interpolated_curve(
            t, traj[:,5], interp_N)

        x_dot = self.get_interpolated_curve(
            t, traj[:,6], interp_N)
        y_dot = self.get_interpolated_curve(
            t, traj[:,7], interp_N)
        z_dot = self.get_interpolated_curve(
            t, traj[:,8], interp_N)

        phi_dot = self.get_interpolated_curve(
            t, traj[:,9], interp_N)
        theta_dot = self.get_interpolated_curve(
            t, traj[:,10], interp_N)
        psi_dot = self.get_interpolated_curve(
            t, traj[:,11], interp_N)

        fig, axs = plt.subplots(5, figsize=(12, 10))

        axs[0].set_ylabel('motor thrust (N)')
        axs[0].plot(t_interp,u1, label='T1')
        axs[0].plot(t_interp,u2, label='T2')   
        axs[0].plot(t_interp,u3, label='T3')
        axs[0].plot(t_interp,u4, label='T4')
        axs[0].legend()

        axs[1].set_ylabel('position (m)')
        axs[1].plot(t_interp,x, label='x')
        axs[1].plot(t_interp,y, label='y')
        axs[1].plot(t_interp,z, label='z')
        axs[1].legend()

        axs[2].set_ylabel('orientation (rad)')
        axs[2].plot(t_interp,phi, label='phi')
        axs[2].plot(t_interp,theta, label='theta')
        axs[2].plot(t_interp,psi, label='psi')
        axs[2].legend()

        axs[3].set_ylabel('velocity (m/s)')
        axs[3].plot(t_interp,x_dot,label='x_dot')
        axs[3].plot(t_interp,y_dot,label='y_dot')
        axs[3].plot(t_interp,z_dot,label='z_dot')
        axs[3].legend()

        axs[4].set_ylabel('body angular vel (rad/s)')
        axs[4].plot(t_interp,phi_dot,label='phi_dot')
        axs[4].plot(t_interp,theta_dot,label='theta_dot')
        axs[4].plot(t_interp,psi_dot,label='psi_dot')
        axs[4].legend()

        for ax in axs.flat:
            ax.set(xlabel='time (s)')
            ax.label_outer()

        plt.show()
        return

    
    def get_interpolated_curve(self, Xs: np.ndarray, Ys: np.ndarray, N: int):
        spline_func = make_interp_spline(Xs, Ys)
        interp_x = np.linspace(Xs.min(), Xs.max(), N)
        interp_y = spline_func(interp_x)
        return interp_y


    def delete_compiled_files(self):
        ''' Clean up the acados generated files.
        '''
        try: shutil.rmtree('c_generated_code')
        except: print('failed to delete c_generated_code') 
        
        try: os.remove('acados_ocp_nlp.json')
        except: print('failed to delete acados_ocp_nlp.json')


"""
Installing acados:
    https://docs.acados.org/installation/index.html#windows-10-wsl
Installing python interface:
    https://docs.acados.org/python_interface/index.html
May need to install qpOASES version 3.1 as well.
"""

#!/usr/bin/python3

from acados_template import AcadosOcpSolver, AcadosOcp
import numpy as np
from scipy.linalg import block_diag
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import matplotlib
import time
from typing import List, Tuple
import atexit
import shutil
import os
from qrac.models import Quadrotor, AffineQuadrotor,\
                        ParameterizedQuadrotor


class NMPC:
    """
    Nonlinear MPC using Acados's OCP solver.
    """

    def __init__(
        self,
        model: Quadrotor,
        Q: np.ndarray,
        R: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
        time_step: float,
        num_nodes: int,
        rti: bool,
        nlp_tol=10**-6,
        nlp_max_iter=20,
        qp_max_iter=20
    ) -> None:
        """
        Initialize the MPC with dynamics from casadi variables,
        Q & R cost matrices, maximum and minimum thrusts, time-step,
        and number of shooting nodes (length of prediction horizon)
        """
        self._nx = model.nx
        self._nu = model.nu
        self._assert(model, Q, R, u_max, u_min, time_step, num_nodes,)
        self._u_avg = (u_min + u_max) / 2
        self._dt = time_step
        self._N = num_nodes
        self._solver = self._init_solver(
            model=model, Q=Q, R=R, u_min=u_min, u_max=u_max, rti=rti,
            nlp_tol=nlp_tol, nlp_max_iter=nlp_max_iter, qp_max_iter=qp_max_iter
        )
        # deleting acados compiled files when script is terminated.
        atexit.register(self._clear_files)


    @property
    def dt(self) -> float:
        return self._dt


    @property
    def n_set(self) -> int:
        return self._N * self._nx


    def get_input(
        self,
        x: np.ndarray,
        xset: np.ndarray,
        timer=False,
    ) -> np.ndarray:
        """
        Get the first control input from the optimization.
        """
        self._solve(x, xset, timer)
        nxt_ctrl = np.array(self._solver.get(0, "u"))
        return nxt_ctrl


    def get_state(
        self,
        x: np.ndarray,
        xset: np.ndarray,
        timer=False,
    ) -> np.ndarray:
        """
        Get the next state from the optimization.
        """
        self._solve(x, xset, timer)
        nxt_state = np.array(self._solver.get(1, "x"))
        return nxt_state


    def get_trajectory(
        self,
        x: np.ndarray,
        xset: np.ndarray,
        timer=False,
        visuals=False,
    ) -> Tuple[np.ndarray]:
        """
        Get the next state from the optimization.
        """
        self._solve(x, xset, timer)
        opt_xs = np.zeros((self._N, self._nx))
        opt_us = np.zeros((self._N, self._nu))
        for k in range(self._N):
            opt_xs[k] = self._solver.get(k, "x")
            opt_us[k] = self._solver.get(k, "u")
        if visuals:
            self._vis_plots(opt_xs, opt_us)
        return opt_xs, opt_us


    def _solve(
        self,
        x: np.ndarray,
        xset: np.ndarray,
        timer: bool,
    ) -> None:
        """
        Set initial state and setpoint,
        then solve the optimization once.
        """
        if timer: st = time.perf_counter()
        assert x.shape[0] == self._nx
        assert xset.shape[0] == self.n_set

        # bound x to initial state
        self._solver.set(0, "lbx", x)
        self._solver.set(0, "ubx", x)

        # the reference input will be the hover input
        for k in range(self._N):
            yref = np.concatenate(
                (xset[k*self._nx : k*self._nx + self._nx], self._u_avg))
            self._solver.set(k, "yref", yref)
        
        self._solver.solve()
        #self._solver.print_statistics()

        if timer:
            et = time.perf_counter()
            print(f"mpc runtime: {et - st}")


    def _init_solver(
        self,
        model: Quadrotor,
        Q: np.ndarray,
        R: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
        rti: bool,
        nlp_tol: float,
        nlp_max_iter: int,
        qp_max_iter: int
    ) -> AcadosOcpSolver:
        """
        Guide to acados OCP formulation:
        https://github.com/acados/acados/blob/master/docs/problem_formulation/problem_formulation_ocp_mex.pdf
        """
        ny = model.nx + model.nu  # combine x and u into y

        ocp = AcadosOcp()
        ocp.model = model.get_acados_model()
        ocp.dims.N = self._N
        ocp.dims.nx = model.nx
        ocp.dims.nu = model.nu
        ocp.dims.ny = ny
        ocp.dims.nbx = model.nx
        ocp.dims.nbx_0 = ocp.dims.nx
        ocp.dims.nbx_e = ocp.dims.nx
        ocp.dims.nbu = model.nu

        # total horizon in seconds
        ocp.solver_options.tf = self._dt * self._N

        # formulate the default least-squares cost as a quadratic cost
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        # W is a block diag matrix of Q and R costs from standard QP
        ocp.cost.W = block_diag(Q,R)

        # use V coeffs to map x & u to y
        ocp.cost.Vx = np.zeros((ny, model.nx))
        ocp.cost.Vx[: model.nx, : model.nx] = np.eye(model.nx)
        ocp.cost.Vu = np.zeros((ny, model.nu))
        ocp.cost.Vu[-model.nu :, -model.nu :] = np.eye(model.nu)

        # Initialize reference trajectory (will be overwritten)
        ocp.cost.yref = np.zeros(ny)

        # Initial state (will be overwritten)
        ocp.constraints.x0 = np.zeros(model.nx)

        # control input constraints (square of motor freq)
        ocp.constraints.idxbu = np.arange(model.nu)
        ocp.constraints.lbu = u_min
        ocp.constraints.ubu = u_max

        # state constraints: yaw
        ocp.constraints.idxbx = np.array(
            [2, 3, 4, 5,]
        )
        ocp.constraints.lbx = np.array(
            [0, -np.pi/2, -np.pi/2, 0,]
        )
        ocp.constraints.ubx = np.array(
            [10**10, np.pi/2, np.pi/2, 2*np.pi,]
        )
        ocp.constraints.idxbx_e = ocp.constraints.idxbx
        ocp.constraints.lbx_e = ocp.constraints.lbx
        ocp.constraints.ubx_e = ocp.constraints.ubx

        # partial condensing HPIPM is fastest:
        # https://cdn.syscop.de/publications/Frison2020a.pdf
        ocp.solver_options.hpipm_mode = "SPEED_ABS"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.qp_solver_iter_max = qp_max_iter
        ocp.solver_options.nlp_solver_max_iter = nlp_max_iter
        ocp.solver_options.nlp_solver_tol_stat = nlp_tol
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.print_level = 0

        ocp.code_export_directory = "mpc_c_code"
        name = "acados_mpc.json"

        if rti:
            ocp.solver_options.nlp_solver_type = "SQP_RTI"
            solver = AcadosOcpSolver(ocp, json_file=name)
            solver.options_set("rti_phase", 0)
        else:
            ocp.solver_options.nlp_solver_type = "SQP"
            solver = AcadosOcpSolver(ocp,  json_file=name)
        return solver


    def _vis_plots(
        self,
        traj: np.ndarray,
        ctrl_inp: np.ndarray,
    ) -> None:
        """
        Display the series of control inputs
        and trajectory over prediction horizon.
        """
        fig, axs = plt.subplots(5, figsize=(11, 9))
        interp_N = 1000
        t = self._dt * np.arange(self._N)

        legend = ["u1", "u2", "u3", "u4"]
        self._plot_trajectory(
            axs[0], ctrl_inp, t, interp_N, legend,
            "motor thrusts (N)",
        )
        legend = ["x", "y", "z"]
        self._plot_trajectory(
            axs[1], traj[:, 0:3], t, interp_N, legend,
            "position (m)"
        )
        legend = ["roll", "pitch", "yaw"]
        self._plot_trajectory(
            axs[2], traj[:, 3:6], t, interp_N, legend,
            "orientation (rad)"
        )
        legend = ["xdot", "ydot", "zdot"]
        self._plot_trajectory(
            axs[3], traj[:, 6:9], t, interp_N, legend,
            "velocity (m/s)"
        )
        legend = ["roll rate", "pitch rate", "yaw rate"]
        self._plot_trajectory(
            axs[4], traj[:, 9:12], t, interp_N, legend,
            "body frame ang vel (rad/s)",
        )

        for ax in axs.flat:
            ax.set(xlabel="time (s)")
            ax.label_outer()
        plt.show()


    def _plot_trajectory(
        self,
        ax: matplotlib.axes,
        traj: np.ndarray,
        Xs: np.ndarray,
        interp_N: int,
        legend: List[str],
        ylabel: str,
    ) -> None:
        ax.set_ylabel(ylabel)
        for i in range(traj.shape[1]):
            x_interp = self._get_interpolation(Xs, Xs, interp_N)
            y_interp = self._get_interpolation(Xs, traj[:, i], interp_N)
            ax.plot(x_interp, y_interp, label=legend[i])
        ax.legend()


    def _get_interpolation(
        self,
        Xs: np.ndarray,
        Ys: np.ndarray,
        N: int,
    ) -> np.ndarray:
        spline_func = make_interp_spline(Xs, Ys)
        interp_x = np.linspace(Xs.min(), Xs.max(), N)
        interp_y = spline_func(interp_x)
        return interp_y


    def _assert(
        self,
        model: Quadrotor,
        Q: np.ndarray,
        R: np.ndarray,
        u_max: np.ndarray,
        u_min: np.ndarray,
        time_step: float,
        num_nodes: int,
    ) -> None:
        if type(model) != Quadrotor \
            and type(model) !=AffineQuadrotor\
            and type(model) != ParameterizedQuadrotor:
                raise TypeError(
                    "The inputted model must be of type 'Quadrotor'!")
        if type(Q) != np.ndarray:
            raise TypeError(
                "Please input the cost matrix as a numpy array!")
        if type(R) != np.ndarray:
            raise TypeError(
                "Please input the cost matrix as a numpy array!")
        if Q.shape != (self._nx, self._nx):
            raise ValueError(
                f"Please input the state cost matrix as a {self._nx}-by-{self._nx} array!")
        if R.shape != (self._nu, self._nu):
            raise ValueError(
                f"Please input the control cost matrix as a {self._nu}-by-{self.nu} array!")
        if len(u_max) != self._nu or len(u_min) != self._nu:
            raise ValueError(
                f"Please input the minimum or maximum control input as vector of length {self._nu}!")
        if type(time_step) != int and type(time_step) != float:
            raise ValueError(
                "Please input the control loop step as an integer or float!")
        if type(num_nodes) != int:
            raise ValueError(
                "Please input the number of shooting nodes as an integer!")


    def _clear_files(self) -> None:
        """
        Clean up the acados generated files.
        """
        try:
            shutil.rmtree("mpc_c_code")
        except:
            print("failed to delete mpc_c_code")
        try:
            os.remove("acados_mpc.json")
        except:
            print("failed to delete acados_mpc.json")

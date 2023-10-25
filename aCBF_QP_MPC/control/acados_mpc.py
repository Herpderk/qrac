"""
Installing acados:
    https://docs.acados.org/installation/index.html#windows-10-wsl
Installing python interface:
    https://docs.acados.org/python_interface/index.html
May need to install qpOASES version 3.1 as well.
"""

#!/usr/bin/python3

from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import matplotlib
import time
from typing import List, Tuple
import atexit
import shutil
import os


class AcadosMpc:
    """
    Nonlinear MPC using Acados's OCP solver.
    """

    def __init__(
        self,
        model: AcadosModel,
        Q: np.ndarray,
        R: np.ndarray,
        u_max: np.ndarray,
        u_min: np.ndarray,
        time_step: float,
        num_nodes: int,
    ) -> None:
        """
        Initialize the MPC with dynamics as casadi namespace,
        Q & R cost matrices, maximum and minimum thrusts, time-step,
        and number of shooting nodes (length of prediction horizon)
        """
        self._nx, self._nu = self._get_dims(model)

        assert type(model) == AcadosModel
        assert Q.shape == (self._nx, self._nx)
        assert R.shape == (self._nu, self._nu)
        assert len(u_max) == self._nu
        assert len(u_min) == self._nu
        assert type(time_step) == int or type(time_step) == float
        assert type(num_nodes) == int

        self._dt = time_step
        self._N = num_nodes
        self._u_min = u_min
        self._solver = self._init_solver(model, Q, R, u_max, u_min)

        # deleting acados compiled files when script is terminated.
        atexit.register(self._clear_files)


    @property
    def dt(self) -> float:
        return self._dt


    def get_input(
        self,
        x0: np.ndarray,
        x_set: np.ndarray,
        timer=False,
    ) -> np.ndarray:
        """
        Get the first control action from the optimization.
        """
        self._solve(x0, x_set, timer)
        nxt_ctrl = self._solver.get(0, "u")
        return nxt_ctrl


    def get_state(
        self,
        x0: np.ndarray,
        x_set: np.ndarray,
        timer=False,
        visuals=False,
    ) -> np.ndarray:
        """
        Get the next state from the optimization.
        """
        self._solve(x0, x_set, timer)
        nxt_state = self._solver.get(1, "x")

        if visuals:
            opt_us = np.zeros((self._N, self._nu))
            opt_xs = np.zeros((self._N, self._nx))
            for k in range(self._N):
                opt_us[k] = self._solver.get(k, "u")
                opt_xs[k] = self._solver.get(k, "x")
            self._vis_plots(opt_us, opt_xs)
        return nxt_state


    def _solve(
        self,
        x0: np.ndarray,
        x_set: np.ndarray,
        timer: bool,
    ) -> None:
        """
        Set initial state and setpoint,
        then solve the optimization once.
        """
        st = time.perf_counter()
        assert len(x0) == self._nx
        assert len(x_set) == self._nx

        # bound x0 to initial state
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        # the reference input will be the hover input
        y_ref = np.concatenate((x_set, self._u_min))
        for k in range(self._N):
            self._solver.set(k, "yref", y_ref)

        # solve for the next ctrl input
        self._solver.solve()
        if timer:
            print(f"mpc runtime: {time.perf_counter() - st}")


    def _get_dims(
        self,
        model: AcadosModel,
    ) -> Tuple[int]:
        """
        Acados model format:
        f_imp_expr/f_expl_expr, x, xdot, u, name
        """
        assert type(model) == AcadosModel
        nx = model.x.shape[0]
        nu = model.u.shape[0]
        return nx, nu


    def _init_solver(
        self,
        model: AcadosModel,
        Q: np.ndarray,
        R: np.ndarray,
        u_max: np.ndarray,
        u_min: np.ndarray,
    ) -> AcadosOcpSolver:
        """
        Guide to acados OCP formulation:
        https://github.com/acados/acados/blob/master/docs/problem_formulation/problem_formulation_ocp_mex.pdf
        """
        ny = self._nx + self._nu  # combine x and u into y

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self._N
        ocp.dims.nx = self._nx
        ocp.dims.nu = self._nu
        ocp.dims.ny = ny
        ocp.dims.nbx_0 = self._nx
        ocp.dims.nbu = self._nu
        ocp.dims.nbx = 3  # number of states being constrained

        # total horizon in seconds
        ocp.solver_options.tf = self._dt * self._N

        # formulate the default least-squares cost as a quadratic cost
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        # W is a block diag matrix of Q and R costs from standard QP
        ocp.cost.W = np.block([
                [Q, np.zeros((self._nx, self._nu))],
                [np.zeros((self._nu, self._nx)), R],
            ])

        # use V coeffs to map x & u to y
        ocp.cost.Vx = np.zeros((ny, self._nx))
        ocp.cost.Vx[: self._nx, : self._nx] = np.eye(self._nx)
        ocp.cost.Vu = np.zeros((ny, self._nu))
        ocp.cost.Vu[-self._nu :, -self._nu :] = np.eye(self._nu)

        # Initialize reference trajectory (will be overwritten)
        ocp.cost.yref = np.zeros(ny)

        # Initial state (will be overwritten)
        ocp.constraints.x0 = np.zeros(self._nx)

        # control input constraints (square of motor freq)
        ocp.constraints.lbu = u_min
        ocp.constraints.ubu = u_max
        ocp.constraints.idxbu = np.arange(self._nu)
        '''
        # state constraints: z, roll, pitch, yaw
        ocp.constraints.lbx = np.array(
            [-10, -10, 0,]
        )
        ocp.constraints.ubx = np.array(
            [10, 10, 10,]
        )
        ocp.constraints.idxbx = np.array(
            [0, 1, 2,]
        )'''

        # not sure what this is, but this paper say partial condensing HPIPM
        # is fastest: https://cdn.syscop.de/publications/Frison2020a.pdf
        ocp.solver_options.hpipm_mode = "SPEED_ABS"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_iter_max = 50
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.print_level = 0

        # compile acados ocp
        solver = AcadosOcpSolver(ocp)
        return solver


    def _vis_plots(
        self,
        ctrl_inp: np.ndarray,
        traj: np.ndarray,
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
            "body frame angular velocity (rad/s)",
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


    def _clear_files(self) -> None:
        """
        Clean up the acados generated files.
        """
        try:
            shutil.rmtree("c_generated_code")
        except:
            print("failed to delete c_generated_code")

        try:
            os.remove("acados_ocp_nlp.json")
        except:
            print("failed to delete acados_ocp_nlp.json")

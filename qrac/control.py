"""
Installing acados:
    https://docs.acados.org/installation/index.html#windows-10-wsl
Installing python interface:
    https://docs.acados.org/python_interface/index.html
May need to install qpOASES version 3.1 as well.
"""

#!/usr/bin/python3

import multiprocessing as mp
import time
from typing import List, Tuple
import atexit
import shutil
import os
import casadi as cs
import numpy as np
from scipy.linalg import block_diag, expm
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import matplotlib
from acados_template import AcadosOcpSolver, AcadosOcp, ocp_get_default_cmake_builder
from qrac.models import Quadrotor, AffineQuadrotor, ParameterizedQuadrotor, L1Quadrotor


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
        uset=np.array([]),
        p=np.array([]),
        timer=False,
    ) -> np.ndarray:
        """
        Get the first control input from the optimization.
        """
        self._solve(x=x, xset=xset, uset=uset, p=p, timer=timer)
        nxt_ctrl = np.array(self._solver.get(0, "u"))
        return nxt_ctrl

    def get_state(
        self,
        x: np.ndarray,
        xset: np.ndarray,
        uset=np.array([]),
        p=np.array([]),
        timer=False,
    ) -> np.ndarray:
        """
        Get the next state from the optimization.
        """
        self._solve(x=x, xset=xset, uset=uset, p=p, timer=timer)
        nxt_state = np.array(self._solver.get(1, "x"))
        return nxt_state

    def get_trajectory(
        self,
        x: np.ndarray,
        xset: np.ndarray,
        uset=np.array([]),
        p=np.array([]),
        timer=False,
        visuals=False,
    ) -> Tuple[np.ndarray]:
        """
        Get the next state from the optimization.
        """
        self._solve(x=x, xset=xset, uset=uset, p=p, timer=timer)
        opt_xs = np.zeros((self._N-1, self._nx))
        opt_us = np.zeros((self._N-1, self._nu))
        for k in range(self._N-1):
            opt_xs[k] = self._solver.get(k, "x")
            opt_us[k] = self._solver.get(k, "u")
        if visuals:
            self._vis_plots(opt_xs, opt_us)
        return opt_xs, opt_us

    def _solve(
        self,
        x: np.ndarray,
        xset: np.ndarray,
        p: np.ndarray,
        timer: bool,
        uset=np.array([]),
    ) -> None:
        """
        Set initial state and setpoint,
        then solve the optimization once.
        """
        if timer: st = time.perf_counter()
        if not len(uset):
            uset = np.tile(self._u_avg, self._N)
        assert x.shape[0] == self._nx
        assert xset.shape[0] == self.n_set
        assert uset.shape[0] == self._nu * self._N

        # bound x to initial state
        self._solver.set(0, "lbx", x)
        self._solver.set(0, "ubx", x)

        for k in range(self._N-1):
            yref = np.hstack((
                xset[k*self._nx : k*self._nx + self._nx],
                uset[k*self._nu : k*self._nu + self._nu]
            ))
            self._solver.set(k, "yref", yref)
        self._solver.set(0, "p", p)
        self._solver.set(self._N-1, "y_ref", xset[(self._N-1)*self._nx:])
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
        ocp.dims.N = self._N-1
        ocp.dims.nx = model.nx
        ocp.dims.nu = model.nu
        ocp.dims.ny = ny
        ocp.dims.nbx = model.nx
        ocp.dims.nbx_0 = ocp.dims.nx
        ocp.dims.nbx_e = ocp.dims.nx
        ocp.dims.nbu = model.nu

        # total horizon in seconds
        ocp.solver_options.tf = self._dt * (self._N-1)

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
        
        # terminal cost
        ocp.cost.W_e = Q
        ocp.cost.Vx_e = np.eye(model.nx)

        # Initialize reference trajectory (will be overwritten)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(model.nx)

        # init parameter vector
        ocp.parameter_values = np.zeros(model.np)

        # Initial state (will be overwritten)
        ocp.constraints.x0 = np.zeros(model.nx)

        # control input constraints (square of motor freq)
        ocp.constraints.idxbu = np.arange(model.nu)
        ocp.constraints.lbu = u_min
        ocp.constraints.ubu = u_max

        # state constraints: yaw
        ocp.constraints.idxbx = np.array(
            [2, 3, 4, 5, 6]
        )
        ocp.constraints.lbx = np.array(
            [0, -1, -1, -1, -1,]
        )
        ocp.constraints.ubx = np.array(
            [10**10, 1, 1, 1, 1,]
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
        builder = ocp_get_default_cmake_builder()

        if rti:
            ocp.solver_options.nlp_solver_type = "SQP_RTI"
            solver = AcadosOcpSolver(ocp, json_file=name, cmake_builder=builder)#, build=False, generate=False)
            solver.options_set("rti_phase", 0)
        else:
            ocp.solver_options.nlp_solver_type = "SQP"
            solver = AcadosOcpSolver(ocp,  json_file=name, cmake_builder=builder)
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
        t = self._dt * np.arange(self._N-1)

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
        legend = ["q0", "q1", "q2", "q3"]
        self._plot_trajectory(
            axs[2], traj[:, 3:7], t, interp_N, legend,
            "orientation (rad)"
        )
        legend = ["xdot", "ydot", "zdot"]
        self._plot_trajectory(
            axs[3], traj[:, 7:10], t, interp_N, legend,
            "velocity (m/s)"
        )
        legend = ["roll rate", "pitch rate", "yaw rate"]
        self._plot_trajectory(
            axs[4], traj[:, 10:13], t, interp_N, legend,
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
        if not isinstance(Q, np.ndarray):
            raise TypeError(
                "Please input the cost matrix as a numpy array!")
        if not isinstance(R, np.ndarray):
            raise TypeError(
                "Please input the cost matrix as a numpy array!")
        if Q.shape != (self._nx, self._nx):
            raise ValueError(
                f"Please input the state cost matrix as a {self._nx}-by-{self._nx} array!")
        if R.shape != (self._nu, self._nu):
            raise ValueError(
                f"Please input the control cost matrix as a {self._nu}-by-{self._nu} array!")
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


def npify(arr_like) -> np.ndarray:
    return np.array(arr_like[:])


class AdaptiveNMPC():
    def __init__(
        self,
        model: Quadrotor,
        estimator,
        Q: np.ndarray,
        R: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
        time_step: float,
        num_nodes: int,
        real_time: bool,
        rti: bool,
        nlp_tol=10**-6,
        nlp_max_iter=10,
        qp_max_iter=10,
    ) -> None:
        self._nx = model.nx
        self._nu = model.nu
        self._N = num_nodes
        self._rt = real_time

        self._est = estimator
        if estimator.is_nonlinear:
            model_aug = ParameterizedQuadrotor(model)
        else:
            model_aug = AffineQuadrotor(model)
        self._mpc = NMPC(
            model=model_aug, Q=Q, R=R,
            u_min=u_min, u_max=u_max,
            time_step=time_step,
            num_nodes=num_nodes, rti=rti,
            nlp_tol=nlp_tol, nlp_max_iter=nlp_max_iter,
            qp_max_iter=qp_max_iter
        )

        self._p = mp.Array("f", model_aug.get_parameters())
        self._x = mp.Array("f", np.zeros(model.nx))
        self._u = mp.Array("f", np.zeros(self._nu))
        self._timer = mp.Value("b", False)
        if real_time:
            self._run_flag = mp.Value("b", True)

    @property
    def dt(self) -> float:
        return self._mpc.dt

    @property
    def n_set(self) -> int:
        return self._N * self._nx

    def start(self) -> None:
        if not self._rt:
            print("Cannot call 'start' outside of real-time mode!")
        else:
            proc = mp.Process(target=self._param_proc, args=[])
            proc.start()

    def stop(self) -> None:
        if not self._rt:
            print("Cannot call 'stop' outside of real-time mode!")
        else:
            self._run_flag.value = False

    def get_input(
        self,
        x: np.ndarray,
        xset: np.ndarray,
        uset=np.array([]),
        timer=False,
    ) -> np.ndarray:
        self._x[:] = x
        self._timer.value = timer
        if not self._rt:
            self._get_param()

        self._u[:] = self._mpc.get_input(
            x=x, xset=xset, uset=uset, p=npify(self._p), timer=timer
        )
        return npify(self._u)

    def _param_proc(self) -> None:
        param_proc = mp.Process(target=self._run_param_est)
        param_proc.start()
        param_proc.join()
        print("\nParameter Estimator successfully stopped.")

    def _run_param_est(self) -> None:
        st = time.perf_counter()
        while self._run_flag.value:
            et = time.perf_counter()
            if et - st >= self.dt:
                st = et
                self._get_param()

    def _get_param(self) -> None:
        param = self._est.get_param(
            x=npify(self._x),
            u=npify(self._u),
            param=npify(self._p),
            timer=self._timer.value
        )
        self._p[:] = param


class L1Augmentation():
    def __init__(
        self,
        model: Quadrotor,
        control_ref,
        adapt_gain: float,
        bandwidth: float,
        opt_horizon=0,
        adapt_gain_min=None,
        adapt_gain_max=None,
        bandwidth_min=None,
        bandwidth_max=None,
        u_min=None,
        u_max=None,
        rti=None,
        nlp_tol=None,
        nlp_max_iter=None,
        qp_max_iter=None,
    ) -> None:
        self._assert(model, control_ref, adapt_gain, bandwidth,)
        model_aug = L1Quadrotor(model)
        self._z_idx = model.nx - model_aug.nz
        self._nz = model_aug.nz
        self._nm = model.nu
        self._num = model_aug.n_um
        self._nx = model.nx

        self._l1p = np.array([adapt_gain, bandwidth])
        self._Am = -1*np.eye(6)
        self._adapt_exp, self._adapt_mat = \
            self._get_l1_const(control_ref.dt)
        self._f, self._g_m, self._g_um = \
            L1Quadrotor(model).get_predictor_funcs()
        #self._f, self._g_m, self._g_um = self._get_dynamics_funcs(model)
        
        self._x = np.zeros(model.nx)
        self._xset = np.zeros(control_ref.n_set)
        self._d_m = np.zeros(self._nm)
        self._d_um = np.zeros(self._num)
        self._ul1 = np.zeros(model.nu)
        self._u = np.zeros(model.nu)
        self._z = np.zeros(self._nz)

        self._ctrl_ref = control_ref
        self._dt = control_ref.dt
        if not (
            opt_horizon > 0
            and adapt_gain_min != None and adapt_gain_max != None
            and bandwidth_min != None and bandwidth_max != None and rti != None
            and nlp_tol != None and nlp_max_iter != None and qp_max_iter != None
        ):
            self._l1_opt = None
        else:
            self._l1_opt = L1Optimizer(
                model=model, u_min=u_min, u_max=u_max,
                a_gain=adapt_gain, bandwidth=bandwidth,
                a_gain_min=adapt_gain_min, a_gain_max=adapt_gain_max,
                bandwidth_min=bandwidth_min, bandwidth_max=bandwidth_max,
                Ts=self._dt, M=opt_horizon, Am=self._Am, rti=rti,
                nlp_tol=nlp_tol, nlp_max_iter=nlp_max_iter, qp_max_iter=qp_max_iter
            )

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def n_set(self) -> Tuple[int]:
        return self._ctrl_ref.n_set

    def get_input(
        self,
        x: np.ndarray,
        xset: np.ndarray,
        uset=[],
        timer=False,
    ) -> np.ndarray:
        assert len(x) == self._nx
        assert len(xset) == self._ctrl_ref.n_set
        uref = self._ctrl_ref.get_input(x=x, xset=xset, uset=uset, timer=timer)
        ul1 = self._get_l1_input(x=x, uref=uref, timer=timer)
        self._u = uref + ul1
        print(f"ul1: {ul1}")
        return self._u

    def _get_l1_input(
        self,
        x: np.ndarray,
        uref: np.ndarray,
        timer: bool
    ) -> None:
        st = time.perf_counter()
        self._optimization(x=x)
        d_m, d_um = self._adaptation(x=x)
        ul1 = self._control_law(d_m=d_m)
        self._predictor(
            x=x, uref=uref, ul1=ul1,
            d_m=d_m, d_um=d_um
        )   
        if timer:
            print(f"L1 runtime: {time.perf_counter() - st}")
        return ul1

    def _optimization(
        self,
        x: np.ndarray,
    ) -> None:
        if self._l1_opt != None:
            self._l1p = self._l1_opt.get_l1_parameters(
                l1p=self._l1p, zpred=self._z, x=x,
                u=self._u, ul1=self._ul1,
            )
            print(f"\noptimized L1 parameters: {self._l1p}\n")

    def _adaptation(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        z_err = np.array(
            self._z - x[self._z_idx : self._z_idx+self._nz]
        )
        g_m = np.array(self._g_m(x))
        g_um = np.array(self._g_um(x))

        I = np.eye(self._nz)
        G = np.hstack((g_m, g_um))
        mu = self._adapt_exp @ z_err
        adapt = self._l1p[0] * -I @ np.linalg.inv(G) @ self._adapt_mat @ mu
        d_m = adapt[: self._nm]
        d_um = adapt[self._nm :]
        return d_m, d_um

    def _control_law(
        self,
        d_m: np.ndarray
    ) -> None:
        self._ul1 = np.exp(-self._l1p[1]*self._dt) * self._ul1 \
            - (1-np.exp(-self._l1p[1]*self._dt))*d_m
        return self._ul1

    def _predictor(
        self,
        x: np.ndarray,
        uref: np.ndarray,
        ul1: np.ndarray,
        d_m: np.ndarray,
        d_um: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z_err = np.array(
            self._z - x[self._z_idx : self._z_idx+self._nz]
        )
        f = np.array(self._f(x, uref)).flatten()
        g_m = np.array(self._g_m(x))
        g_um = np.array(self._g_um(x))
        self._z += self._dt * (
            self._Am @ z_err \
            + f + g_m@(ul1 + d_m) + g_um@d_um
        )

    def _get_l1_const(
        self,
        time_step: float
    ) -> Tuple[np.ndarray]:
        adapt_exp = expm(self._Am*time_step)
        adapt_mat = np.linalg.inv(self._Am) @ (adapt_exp - np.eye(self._nz))
        return adapt_exp, adapt_mat

    def _get_dynamics_funcs(
        self,
        model: Quadrotor,
    ) -> Tuple[cs.Function]:
        # rotation matrix from body frame to inertial frame
        b1 = model.R[:,0]
        b2 = model.R[:,1]
        b3 = model.R[:,2]

        f = model.xdot[6:12]
        g_m = cs.SX(cs.vertcat(
            b3/model.m @ cs.SX.ones(1,4),
            cs.inv(model.J) @ model.B
        ))
        g_um = cs.SX(cs.vertcat(
            cs.horzcat(b1, b2)/model.m,
            cs.SX.zeros(3,2)
        ))
        f_func = cs.Function("f", [model.x, model.u], [f])
        g_m_func = cs.Function("g_m", [model.x], [g_m])
        g_um_func = cs.Function("g_um", [model.x], [g_um])
        return f_func, g_m_func, g_um_func

    def _assert(
        self,
        model: Quadrotor,
        control_ref,
        adapt_gain: float,
        bandwidth: float,
    ) -> None:
        if type(model) != Quadrotor:
            raise TypeError(
                "The inputted model must be of type 'NonlinearQuadrotor'!")
        try:
            control_ref.get_input
        except AttributeError:
            raise NotImplementedError(
                "Please implement a 'get_input' method in your reference controller class!")
        try:
            control_ref.dt
        except AttributeError:
            raise NotImplementedError(
                "Please implement a 'dt' attribute in your controller class!")
        try:
            control_ref.n_set
        except AttributeError:
            raise NotImplementedError(
                "Please implement a 'n_set' attribute in your controller class!")

        if type(adapt_gain) != int and type(adapt_gain) != float:
            raise TypeError(
                "Please input the adaptation gain as an integer or float!")
        if type(bandwidth) != int and type(bandwidth) != float:
            raise TypeError(
                "Please input the bandwidth as an integer or float!")


class L1Optimizer():
    def __init__(
        self,
        model: Quadrotor,
        u_min: np.ndarray,
        u_max: np.ndarray,
        a_gain: float,
        bandwidth: float,
        a_gain_min: float,
        a_gain_max: float,
        bandwidth_min: float,
        bandwidth_max:float,
        Am: np.ndarray,
        Ts: float,
        M: int,
        rti: bool,
        nlp_tol: float,
        nlp_max_iter: int,
        qp_max_iter: int,
    ):
        model_aug = L1Quadrotor(model)
        self._Ts = Ts
        self._M = M
        self._nx = model.nx
        self._nu = model.nu
        self._nz = model_aug.nz
        self._np = model_aug.nx + 2*model.nu
        self._ny = model_aug.ny
        self._solver = self._init_solver(
            model=model_aug, u_min=u_min, u_max=u_max,
            a_gain_min=a_gain_min, a_gain_max=a_gain_max,
            bandwidth_min=bandwidth_min, bandwidth_max=bandwidth_max,
            time_step=Ts, M=M, Am=Am, rti=rti, nlp_tol=nlp_tol,
            nlp_max_iter=nlp_max_iter, qp_max_iter=qp_max_iter
        )
        self._nl1p = model_aug.nu

        # make sure model_aug.get_discrete_model() is called
        # before setting model_aug dimensions
        self._l1p_init = np.array([a_gain, bandwidth])
        self._l1p = self._l1p_init
        self._l1pl = np.array([a_gain_min, bandwidth_min])
        self._l1pu = np.array([a_gain_max, bandwidth_max])
        #self._x = np.zeros((1, model.nx))
        self._x = np.zeros((self._M, model.nx))
        self._zpred = np.zeros((self._M, model_aug.nz))
        self._u = np.zeros((self._M, model.nu))
        self._ul1 = np.zeros((self._M+1, model.nu))

    def get_l1_parameters(
        self,
        l1p: np.ndarray,
        zpred: np.ndarray,
        x: np.ndarray,
        u: np.ndarray,
        ul1: np.ndarray,
    ) -> np.ndarray:
        self._l1p = l1p
        self._update_horizon(
            zpred=zpred, x=x,
            u=u, ul1=ul1
        )
        '''
        if self._ul1.shape[0] < self._M+1:
            pass
            print(self._x)
            print(self._ul1)
        else:'''
        self._solve()
        return self._l1p

    def _update_horizon(
        self,
        zpred: np.ndarray,
        x: np.ndarray,
        u: np.ndarray,
        ul1: np.ndarray,
    ) -> None:
        i0 = 1
        i1 = 1
        '''
        if self._u.shape[0] >= self._M:
            i0 = 1
        if self._ul1.shape[0] >= self._M+1:
            i1 = 1
        '''
            
        self._zpred = np.vstack(
            (self._zpred[i0:self._M, :], zpred)
        )
        self._x = np.vstack(
            (self._x[i0:self._M, :], x)
        )
        self._u = np.vstack(
            (self._u[i0:self._M, :], u)
        )
        self._ul1 = np.vstack(
            (self._ul1[i1:self._M+1, :], ul1)
        )

    def _solve(self) -> None:
        p = np.zeros(self._M*self._np)
        x = np.zeros(self._M*self._ny)
        y_ref = np.zeros(self._M*self._ny + self._nl1p)

        y_ref[-self._nl1p :] = self._l1p
        for k in range(self._M):
            p[k*self._np : (k+1)*self._np] = np.hstack(
                (self._x[k], self._ul1[k:k+2].flatten())
            )
            x[k*self._ny : (k+1)*self._ny] = np.hstack(
                (self._zpred[k], self._u[k])
            )
            y_ref[k*self._ny : (k+1)*self._ny] = np.zeros(self._nz + self._nu)
            '''
            np.hstack(
                (self._x[k, self._nx-self._nz:self._nx], np.zeros(self._nu))
            )'''

        self._solver.set(0, "u", self._l1p)
        self._solver.set(0, "lbu", self._l1pl)
        self._solver.set(0, "ubu", self._l1pu)
        self._solver.set(0, "p", p)
        self._solver.set(0, "lbx", x)
        self._solver.set(0, "ubx", x)
        self._solver.set(1, "y_ref", y_ref)
        self._solver.solve()
        #self._solver.print_statistics()
        self._l1p = self._solver.get(0,"u")

    def _init_solver(
        self,
        model: L1Quadrotor,
        u_min: np.ndarray,
        u_max: np.ndarray,
        a_gain_min: float,
        a_gain_max: float,
        bandwidth_min: float,
        bandwidth_max: float,
        time_step: float,
        M: int,
        Am: np.ndarray,
        rti: bool,
        nlp_tol: float,
        nlp_max_iter: int,
        qp_max_iter: int
    ) -> AcadosOcpSolver:
        assert len(u_min) == len(u_max) == self._nu
        ocp = AcadosOcp()
        ocp.model = model.get_discrete_model(
            M=M, Ts=time_step, Am=Am
        )
        ny = model.nx + model.nu  # combine x and u into y
        ocp.dims.N = 2
        ocp.dims.nx = model.nx
        ocp.dims.nu = model.nu
        ocp.dims.ny = ny
        ocp.dims.nbx = model.nx
        ocp.dims.nbx_0 = ocp.dims.nbx
        ocp.dims.nbx_e = ocp.dims.nbx
        ocp.dims.nbu = model.nu

        # total horizon in seconds
        ocp.solver_options.tf = time_step * ocp.dims.N

        # formulate the default least-squares cost as a quadratic cost
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        # construct Q for full state
        R = 10**-10*np.diag(np.ones(model.nu))
        Q = 10**10*np.diag(np.hstack(
            (np.ones(self._nz), np.zeros(self._nu))
        ))
        Q_full = Q
        for k in range(1, self._M):
            Q_full = block_diag(Q_full, Q)

        # W is a block diag matrix of Q and R costs from standard QP
        ocp.cost.W = block_diag(Q_full, R)
        # use V coeffs to map x & u to y
        ocp.cost.Vx = np.zeros((ny, model.nx))
        ocp.cost.Vx[: model.nx, : model.nx] = np.eye(model.nx)
        ocp.cost.Vu = np.zeros((ny, model.nu))
        ocp.cost.Vu[-model.nu :, -model.nu :] = np.eye(model.nu)

        # terminal cost
        ocp.cost.W_e = Q_full
        ocp.cost.Vx_e = np.eye(model.nx)

        # init reference trajectory (will be overwritten)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(model.nx)

        # init parameter vector
        ocp.parameter_values = np.zeros(model.np)

        # Initial state (will be overwritten)
        ocp.constraints.x0 = np.zeros(model.nx)

        # l1 parameter constraints
        ocp.constraints.idxbu = np.arange(model.nu)
        
        ocp.constraints.lbu = np.array([a_gain_min, bandwidth_min])
        ocp.constraints.ubu = np.array([a_gain_max, bandwidth_max])
        '''
        ocp.constraints.lbu = np.array([bandwidth_min])
        ocp.constraints.ubu = np.array([bandwidth_max])
        '''

        # control input constraints
        ocp.constraints.idxbx_e = np.zeros(self._nu * self._M)
        ocp.constraints.lbx_e = np.zeros(self._nu * self._M)
        ocp.constraints.ubx_e = np.zeros(self._nu * self._M)
        nz = self._nz
        nu = self._nu
        for k in range(self._M):
            ocp.constraints.idxbx_e[k*nu : (k+1)*nu] = \
                (k+1)*nz + np.arange(nu)
            ocp.constraints.lbx_e[k*nu : (k+1)*nu] = \
                u_min * np.ones(nu)
            ocp.constraints.ubx_e[k*nu : (k+1)*nu] = \
                u_max * np.ones(nu)

        # partial condensing HPIPM is fastest:
        # https://cdn.syscop.de/publications/Frison2020a.pdf
        ocp.solver_options.hpipm_mode = "ROBUST"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.qp_solver_iter_max = qp_max_iter
        ocp.solver_options.nlp_solver_max_iter = nlp_max_iter
        ocp.solver_options.nlp_solver_tol_stat = nlp_tol
        ocp.solver_options.hessian_approx = "EXACT" # "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_ext_qp_res = 1

        ocp.code_export_directory = "l1opt_c_code"
        name = "acados_l1opt.json"
        builder = ocp_get_default_cmake_builder()

        if rti:
            ocp.solver_options.nlp_solver_type = "SQP_RTI"
            solver = AcadosOcpSolver(ocp, json_file=name, cmake_builder=builder)#, build=False, generate=False)
            solver.options_set("rti_phase", 0)
        else:
            ocp.solver_options.nlp_solver_type = "SQP"
            solver = AcadosOcpSolver(ocp, json_file=name, cmake_builder=builder)
        return solver

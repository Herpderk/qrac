#!/usr/bin/python3

import time
from typing import Tuple
import atexit
import shutil
import os
from acados_template import AcadosOcpSolver, AcadosOcp
import casadi as cs
import numpy as np
from scipy.linalg import block_diag
from qpsolvers.solvers.proxqp_ import proxqp_solve_qp
from qrac.models import Quadrotor, AffineQuadrotor, ParameterizedQuadrotor


class SetMembershipEstimator:
    def __init__(
        self,
        model: Quadrotor,
        estimator,
        param_tol: np.ndarray,
        param_min: np.ndarray,
        param_max: np.ndarray,
        disturb_min: np.ndarray,
        disturb_max: np.ndarray,
        time_step: float,
        qp_tol=10**-6,
        max_iter=20,
    ) -> None:
        self._nx = model.nx
        self._est = estimator
        model_aug = AffineQuadrotor(model)
        self._Fd, self._Gd, = self._discretize(
            model_aug, self._nx, time_step
        )

        self._np = model_aug.np
        self._d_min = time_step*disturb_min
        self._d_max = time_step*disturb_max
        self._p_tol = param_tol
        self._qp_tol = qp_tol
        self._max_iter = max_iter

        self._x = np.zeros(self._nx)
        self._p_min = param_min
        self._p_max = param_max
        self._start = False

    @property
    def is_nonlinear(self) -> bool:
        return False

    def _discretize(
        self,
        model: AffineQuadrotor,
        nx: int,
        dt: float
    ) -> Tuple[cs.Function, cs.Function, cs.Function]:
        Fd = model.x[:nx] + dt*model.F
        Gd = dt*model.G
        Fd_func = cs.Function("Fd_func", [model.x[:nx], model.u], [Fd])
        Gd_func = cs.Function("Gd_func", [model.x[:nx], model.u], [Gd])
        return Fd_func, Gd_func

    def get_param(
        self,
        x: np.ndarray,
        u: np.ndarray,
        param: np.ndarray,
        timer=True
    ) -> np.ndarray:
        st = time.perf_counter()
        p_min, p_max = self._sm_update(x, u)
        p = self._est.get_param(
            x=x, u=u, param=param,
            param_min=p_min, param_max=p_max,
            timer=False
        )
        self._x = x
        print(f"param min: {p_min}")
        print(f"param max: {p_max}")
        if timer:
            et = time.perf_counter()
            print(f"sm runtime: {et - st}")
        return p

    def _sm_update(
        self,
        x: np.ndarray,
        u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._start == False:
            self._start = True
            self._x = x

        p_min = self._p_min
        p_max = self._p_max
        if not (p_max - p_min < self._p_tol).all():
            sol_min = np.zeros(self._np)
            sol_max = np.zeros(self._np)

            for i in range(self._np):
                if p_max[i] - p_min[i] > self._p_tol[i]:
                    sol_min[i] = self._solve_lp(
                        idx=i, x=x, xprev=self._x,
                        u=u, is_max=False
                    )
                    sol_max[i] = self._solve_lp(
                        idx=i, x=x, xprev=self._x,
                        u=u, is_max=True
                    )
                else:
                    sol_min[i] = p_min[i]
                    sol_max[i] = p_max[i]

            self._p_min = np.maximum(sol_min, p_min)
            self._p_max = np.minimum(sol_max, p_max)
        return self._p_min, self._p_max

    def _solve_lp(
        self,
        idx: int,
        x: np.ndarray,
        xprev: np.ndarray,
        u: np.ndarray,
        is_max: bool
    ) -> float:
        P = np.zeros((self._np, self._np))
        q = np.zeros(self._np)
        if is_max:
            q[idx] = -1.0
        else:
            q[idx] = 1.0

        Fd = np.array(self._Fd(xprev, u)).flatten()
        Gd = np.array(self._Gd(xprev, u))
        G = np.block([[Gd], [-Gd]])
        h = np.round(
            np.block([x-Fd-self._d_min, -x+Fd+self._d_max]), 2
        )

        if is_max:
            p_init = self._p_max
        else:
            p_init = self._p_min
        p_bd = proxqp_solve_qp(
            P=P, q=q, G=G, h=h,
            lb=self._p_min, ub=self._p_max, initvals=p_init,
            verbose=False, backend="dense",
            eps_abs=self._qp_tol, max_iter=self._max_iter,
        )

        try:
            return p_bd[idx]
        except TypeError:
            if is_max:
                return self._p_max[idx]
            else:
                return self._p_min[idx]

    def _update_param_bds(
        self,
        param_min: np.ndarray,
        param_max: np.ndarray,
    ) -> None:
        self._p_min = param_min
        self._p_max = param_max


class LMS():
    def __init__(
        self,
        model: Quadrotor,
        param_min: np.ndarray,
        param_max: np.ndarray,
        update_gain: float,
        time_step: float,
        qp_tol=10**-6,
        max_iter=10,
    ) -> None:
        self._nx = model.nx
        model_aug = AffineQuadrotor(model)
        self._Fd, self._Gd, self._Gd_T = \
            self._get_discrete_dynamics(model_aug, self._nx, time_step)

        self._np = model_aug.np
        self._mu = update_gain
        self._sol_tol = qp_tol
        self._max_iter = max_iter
        self._p_min = param_min
        self._p_max = param_max
        self._x = np.zeros(self._nx)

    @property
    def is_nonlinear(self) -> bool:
        return False

    def _get_discrete_dynamics(
        self,
        model: AffineQuadrotor,
        nx: int,
        dt: float
    ) -> Tuple[cs.Function, cs.Function, cs.Function]:
        Fd = model.x[:nx] + dt*model.F
        Gd = dt*model.G
        Fd_func = cs.Function("Fd_func", [model.x[:nx], model.u], [Fd])
        Gd_func = cs.Function("Gd_func", [model.x[:nx], model.u], [Gd])
        Gd_T_func = cs.Function("Gd_T_func", [model.x[:nx], model.u], [Gd.T])
        return Fd_func, Gd_func, Gd_T_func

    def get_param(
        self,
        x: np.ndarray,
        u: np.ndarray,
        param: np.ndarray,
        param_min=np.array([]),
        param_max=np.array([]),
        timer=True
    ) -> np.ndarray:
        if timer: st = time.perf_counter()
        x_err = x - self._Fd(self._x, u) - self._Gd(self._x, u)@param
        p_lms = param + self._mu*self._Gd_T(self._x, u)@x_err
        p_lms = np.array(p_lms).flatten()

        if param_min.shape[0] == 0:
            param_min = self._p_min
        if param_max.shape[0] == 0:
            param_max = self._p_max
        p_proj = self._solve_proj(
            p=p_lms, p_min=param_min, p_max=param_max
        )
        self._x = x
        print(f"params: {p_proj}\n")
        if timer:
            et = time.perf_counter()
            print(f"LMS runtime: {et - st}")
        print(f"\nparams: {p_proj}\n")
        return p_proj

    def _solve_proj(
        self,
        p: np.ndarray,
        p_min: np.ndarray,
        p_max: np.ndarray
    ) -> np.ndarray:
        P = np.eye(self._np)
        q = np.zeros(self._np)
        lb = p_min - p
        ub = p_max - p
        p_err = proxqp_solve_qp(
            P=P, q=q, lb=lb, ub=ub,
            verbose=False, backend="dense",
            eps_abs=self._sol_tol, max_iter=self._max_iter,
        )
        try:
            p_proj = p_err + p
            return p_proj
        except TypeError:
            return p


class MHE():
    def __init__(
        self,
        model: Quadrotor,
        Q: np.ndarray,
        R: np.ndarray,
        param_min: np.ndarray,
        param_max: np.ndarray,
        disturb_min: np.ndarray,
        disturb_max: np.ndarray,
        time_step: float,
        num_nodes: int,
        rti: bool,
        nlp_tol=10**-6,
        nlp_max_iter=10,
        qp_max_iter=10,
        nonlinear=False,
    ) -> None:
        """
        Q -> weight for params
        R -> weight for disturb
        """
        self._nx = model.nx
        self._nu = model.nu
        self._dt = time_step
        self._N = num_nodes
        self._d_min = disturb_min
        self._d_max = disturb_max
        self._d_avg = (disturb_min + disturb_max) / 2
        self._nl = nonlinear

        if nonlinear:
            model_aug = ParameterizedQuadrotor(model)
        else:
            model_aug = AffineQuadrotor(model)
        model_aug.set_prediction_model()
        self._np = model_aug.np
        assert param_max.shape[0] == param_min.shape[0] == self._np
        self._p_min = param_min
        self._p_max = param_max
        self._p_init = model_aug.get_parameters()

        Q_aug = self._augment_costs(Q)
        self._solver = self._init_solver(
            model=model_aug, Q=Q_aug, R=R,
            p_min=param_min, p_max=param_max,
            d_min=disturb_min, d_max=disturb_max, rti=rti,
            nlp_tol=nlp_tol, nlp_max_iter=nlp_max_iter,
            qp_max_iter=qp_max_iter
        )
        atexit.register(self._clear_files)

        self._x = np.zeros((self._N, self._nx))
        self._u = np.zeros((self._N-1, self._nu))
        self._d = np.zeros((self._N-1, self._nx))


    @property
    def is_nonlinear(self) -> bool:
        return self._nl

    def get_param(
        self,
        x: np.ndarray,
        u: np.ndarray,
        param: np.ndarray,
        param_min=np.array([]),
        param_max=np.array([]),
        timer=True
    ) -> np.ndarray:
        self._solve(
            x=x, u=u, p=param, p_min=param_min,
            p_max=param_max, timer=timer
        )
        #self._solver.print_statistics()
        p = np.array(
            self._solver.get(0,"x")[self._nx : self._nx+self._np]
        )
        print(f"params: {p}\n")
        return p

    def _solve(
        self,
        x: np.ndarray,
        u: np.ndarray,
        p: np.ndarray,
        p_min: np.ndarray,
        p_max: np.ndarray,
        timer: bool,
    ) -> np.ndarray:
        if timer: st = time.perf_counter()
        assert x.shape[0] == self._nx
        assert u.shape[0] == self._nu
        assert p.shape[0] == self._np

        # set history of x, u, and d at each stage
        for k in range(self._N-1):
            self._set_stage(
                k=k, x=self._x[k], u=self._u[k], d=self._d[k],
                p=p, p_min=p_min, p_max=p_max
            )

        # set new measurements
        self._set_stage(
            k=self._N-1, x=self._x[self._N-1], u=u,
            p=p, p_min=p_min, p_max=p_max
        )
        self._set_stage(
            k=self._N, x=x,
            p=p, p_min=p_min, p_max=p_max
        )
        self._solver.solve()

        # get the latest disturbance estimate
        # propagate the horizon by 1 step
        self._update_horizon(x=x, u=u)

        if timer:
            et = time.perf_counter()
            print(f"mhe runtime: {et - st}")

    def _set_stage(
        self,
        k: int,
        x: np.ndarray,
        p_min: np.ndarray,
        p_max: np.ndarray,
        p=np.array([]),
        u=np.array([]),
        d=np.array([]),
    ):
        x_aug = np.concatenate((x, p))#self._p_init))
        self._solver.set(k, "x", x_aug)

        if k == self._N-1:
            d = np.zeros(self._nx)
        if k != self._N:
            yref = np.concatenate((x_aug, d))
            self._solver.set(k, "yref", yref)
            self._solver.set(k, "u", d)
            self._solver.set(k, "p", u)

        if k!= 0:
            if p_min.shape[0] == 0:
                p_min = self._p_min
            lbx_aug = np.concatenate((x, p_min))
            self._solver.set(k, "lbx", lbx_aug)

            if p_max.shape[0] == 0:
                p_max = self._p_max
            ubx_aug = np.concatenate((x, p_max))
            self._solver.set(k, "ubx", ubx_aug)

    def _update_horizon(
        self,
        x: np.ndarray,
        u: np.ndarray,
    ):
        self._x = np.block([
            [self._x[1:self._N, :]], [x]
        ])
        self._u = np.block([
            [self._u[1:self._N-1, :]], [u]
        ])
        for k in range(1, self._N):
            self._d[k-1,:] = np.array(
                self._solver.get(k, "u")
            )

    def _init_solver(
        self,
        model: Quadrotor,
        Q: np.ndarray,
        R: np.ndarray,
        p_min: np.ndarray,
        p_max: np.ndarray,
        d_min: np.ndarray,
        d_max: np.ndarray,
        rti: bool,
        nlp_tol: float,
        nlp_max_iter: int,
        qp_max_iter: int
    ) -> AcadosOcpSolver:
        ny = model.nx + model.nu  # combine x and u into y

        ocp = AcadosOcp()
        ocp.model = model.get_acados_model()
        ocp.dims.N = self._N
        ocp.dims.nx = model.nx
        ocp.dims.nu = model.nu
        ocp.dims.ny = ny
        ocp.dims.nbx = model.nx
        ocp.dims.nbx_0 = ocp.dims.nbx
        ocp.dims.nbx_e = ocp.dims.nbx
        ocp.dims.nbu = model.nu

        # total horizon in seconds
        ocp.solver_options.tf = self._dt * self._N

        # formulate the default least-squares cost as a quadratic cost
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        # W is a block diag matrix of Q and R costs from standard QP
        ocp.cost.W = block_diag(Q, R)

        # use V coeffs to map x & u to y
        ocp.cost.Vx = np.zeros((ny, model.nx))
        ocp.cost.Vx[: model.nx, : model.nx] = np.eye(model.nx)
        ocp.cost.Vu = np.zeros((ny, model.nu))
        ocp.cost.Vu[-model.nu :, -model.nu :] = np.eye(model.nu)
        ocp.cost.Vu_0 = ocp.cost.Vu

        # init reference trajectory (will be overwritten)
        ocp.cost.yref = np.zeros(ny)

        # init parameter vector
        ocp.parameter_values = np.zeros(self._nu)

        # control input constraints (disturbance)
        ocp.constraints.idxbu = np.arange(model.nu)
        ocp.constraints.lbu = d_min
        ocp.constraints.ubu = d_max

        # augmented state constraints (will be overwritten)
        ocp.constraints.idxbx = np.arange(model.nx)
        ocp.constraints.lbx = np.concatenate(
            (-np.ones(self._nx), p_min)
        )
        ocp.constraints.ubx = np.concatenate(
            (np.ones(self._nx), p_max)
        )
        ocp.constraints.idxbx_e = np.arange(model.nx)
        ocp.constraints.lbx_e = np.concatenate(
            (-np.ones(self._nx), p_min)
        )
        ocp.constraints.ubx_e = np.concatenate(
            (np.ones(self._nx), p_max)
        )

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
        ocp.solver_options.nlp_solver_ext_qp_res = 1

        ocp.code_export_directory = "mhe_c_code"
        name = "acados_mhe.json"

        if rti:
            ocp.solver_options.nlp_solver_type = "SQP_RTI"
            solver = AcadosOcpSolver(ocp, json_file=name)
            solver.options_set("rti_phase", 0)
        else:
            ocp.solver_options.nlp_solver_type = "SQP"
            solver = AcadosOcpSolver(ocp, json_file=name)
        return solver

    def _augment_costs(
        self,
        Q: np.ndarray
    ) -> np.ndarray:
        Q_aug = block_diag(
            (10**10)*np.eye(self._nx), Q
        )
        return Q_aug

    def _clear_files(self) -> None:
        """
        Clean up the acados generated files.
        """
        try:
            shutil.rmtree("mhe_c_code")
        except:
            print("failed to delete mhe_c_code")
        try:
            os.remove("acados_mhe.json")
        except:
            print("failed to delete acados_mhe.json")

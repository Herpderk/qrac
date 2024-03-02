#!/usr/bin/python3

import casadi as cs
import numpy as np
import multiprocessing as mp
from qpsolvers.solvers.proxqp_ import proxqp_solve_qp
from scipy.linalg import block_diag
from typing import Tuple
import time
from qrac.dynamics import Quadrotor, ParameterAffineQuadrotor
from qrac.control.nmpc import NMPC


class SetMembershipMPC():
    def __init__(
        self,
        model: Quadrotor,
        Q: np.ndarray,
        R: np.ndarray,
        update_gain: float,
        param_tol: np.ndarray,
        param_max: np.ndarray,
        param_min: np.ndarray,
        disturb_min: np.ndarray,
        disturb_max: np.ndarray,
        u_max: np.ndarray,
        u_min: np.ndarray,
        time_step: float,
        param_time_step: float,
        num_nodes: int,
        rti: bool,
        real_time: bool,
    ) -> None:
        self._nx = model.nx
        self._nu = model.nu
        self._dt = param_time_step
        self._N = num_nodes
        self._rt = real_time

        self._nparam = 10
        self._param_idx = 12
        self._d_min = disturb_min*time_step
        self._d_max = disturb_max*time_step
        self._mu = update_gain

        self._lp_max_iter = 10
        self._proj_max_iter = 10
        model_aug = ParameterAffineQuadrotor(model)
        self._param_est = ParameterEstimator(
            model_aug, self._nx, time_step,)

        Q_aug = self._augment_costs(Q)
        self._mpc = NMPC(
            model_aug, Q_aug, R, u_min, u_max,
            time_step, num_nodes, rti,
        )

        self._tol = param_tol
        self._param_min = mp.Array("f", param_min)
        self._param_max = mp.Array("f", param_max)
        self._param = mp.Array("f", model_aug.get_parameters())
        self._x = mp.Array("f", np.zeros(model.nx))
        self._xprev = mp.Array("f", np.zeros(self._nx))
        self._uprev = mp.Array("f", np.zeros(self._nu))
        self._timer = mp.Value("b", False)
        self._start = False
        if real_time:
            self._run_flag = mp.Value("b", True)


    @property
    def dt(self) -> float:
        return self._mpc.dt


    @property
    def n_set(self) -> int:
        return self._N * self._nx


    def start(self) -> None:
        if self._rt:
            proc = mp.Process(target=self._param_proc, args=[])
            proc.start()
            

    def stop(self) -> None:
        if self._rt:
            self._run_flag.value = False


    def get_input(
        self,
        x: np.ndarray,
        xset: np.ndarray,
        timer=False,
    ) -> np.ndarray:
        self._xprev[:] = self._x[:]
        self._x[:] = x
        self._timer.value = timer

        if not self._rt:
            self._update_param()

        x_aug = np.concatenate((x, self._np(self._param)))
        xset_aug = self._augment_xset(xset)
        self._uprev[:] = self._mpc.get_input(x_aug, xset_aug, timer)
        return self._np(self._uprev)


    def _augment_xset(
        self,
        xset: np.ndarray
    ) -> np.ndarray:
        nx = self._nx
        nparam = self._nparam
        xset_aug = np.zeros(self._N * (nx+nparam))
        for k in range(self._N):
            xset_aug[k*(nx+nparam) : k*(nx+nparam) + nx] =\
                xset[k*nx : k*nx + nx]
        return xset_aug


    def _np(
        self,
        arr_like
    ) -> np.ndarray:
        return np.array(arr_like[:])


    def _param_proc(self) -> None:
        param_proc = mp.Process(target=self._run_param_updates)
        param_proc.start()
        param_proc.join()
        print("\nController successfully stopped.")


    def _run_param_updates(self) -> None:
        st = time.perf_counter()
        while self._run_flag.value:
            et = time.perf_counter()
            if et - st >= self._dt:
                st = et
                self._update_param()


    def _update_param(
        self,
    ) -> None:
        st = time.perf_counter()
        param = self._np(self._param)
        x = self._np(self._x)
        xprev = self._np(self._xprev)
        uprev = self._np(self._uprev)
        timer = self._timer.value

        param = self._param_est.lms_update(
            x=x, xprev=xprev, uprev=uprev,
            param=param, mu=self._mu
        )
        param_min, param_max = self._sm_update(x, xprev, uprev)
        param_proj = self._param_est.solve_proj(
            param=param, param_min=param_min, param_max=param_max,
            max_iter=self._proj_max_iter
        )
        self._update_param_vars(param_proj, param_min, param_max)
        print(f"params: {param_proj}")
        if timer:
            print(f"sm and lms runtime: {time.perf_counter() - st}")


    def _sm_update(
        self,
        x: np.ndarray,
        xprev: np.ndarray,
        uprev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._start == False:
            self._start = True
            xprev = x

        param_min = self._np(self._param_min)
        param_max = self._np(self._param_max)
        if not (param_max - param_min < self._tol).all():
            sol_min = np.zeros(self._nparam)
            sol_max = np.zeros(self._nparam)
        
            for i in range(self._nparam):
                if param_max[i] - param_min[i] > self._tol[i]:
                    sol_min[i] = self._param_est.solve_lp(
                        idx=i, x=x, xprev=xprev, uprev=uprev,
                        d_min=self._d_min, d_max=self._d_max,
                        param_min=param_min, param_max=param_max,
                        max_iter=self._lp_max_iter, max=False
                    )
                    sol_max[i] = self._param_est.solve_lp(
                        idx=i, x=x, xprev=xprev, uprev=uprev,
                        d_min=self._d_min, d_max=self._d_max,
                        param_min=param_min, param_max=param_max,
                        max_iter=self._lp_max_iter, max=True
                    )
                else:
                    sol_min[i] = param_min[i]
                    sol_max[i] = param_max[i]
            
            param_min = np.maximum(
                sol_min, param_min
            )
            param_max = np.minimum(
                sol_max, param_max
            )
        return param_min, param_max


    def _update_param_vars(
        self,
        param,
        param_min,
        param_max
    ) -> None:
        self._param[:] = param
        self._param_min[:] = param_min
        self._param_max[:] = param_max


    def _augment_costs(
        self,
        Q: np.ndarray,
    ) -> np.ndarray:
        Q_aug = block_diag(Q, np.zeros((self._nparam, self._nparam)))
        return Q_aug


class ParameterEstimator():
    def __init__(
        self,
        model: ParameterAffineQuadrotor,
        nx: int,
        dt: float,
    ) -> None:
        self._nparam = model.nx - nx
        self._Fd, self._Gd, self._Gd_T = self._get_discrete_dynamics(model, nx, dt)


    def _get_discrete_dynamics(
        self,
        model: ParameterAffineQuadrotor,
        nx: int,
        dt: float
    ):
        Fd = model.x[:nx] + dt*model.F
        Gd = dt*model.G
        Fd_func = cs.Function("Fd_func", [model.x[:nx], model.u], [Fd])
        Gd_func = cs.Function("Gd_func", [model.x[:nx], model.u], [Gd])
        Gd_T_func = cs.Function("Gd_T_func", [model.x[:nx], model.u], [Gd.T])
        return Fd_func, Gd_func, Gd_T_func


    def lms_update(
        self,
        x: np.ndarray,
        xprev: np.ndarray,
        uprev: np.ndarray,
        param: np.ndarray,
        mu: float
    ) -> np.ndarray:
        x_err = x - self._Fd(xprev, uprev) - self._Gd(xprev, uprev)@param
        param_lms = param + mu*self._Gd_T(xprev, uprev)@x_err
        param_lms = np.array(param_lms).flatten()
        return param_lms


    def solve_lp(
        self,
        idx: int,
        x: np.ndarray,
        xprev: np.ndarray,
        uprev: np.ndarray,
        d_min: np.ndarray,
        d_max: np.ndarray,
        param_min: np.ndarray,
        param_max: np.ndarray,
        max_iter: int,
        max: bool
    ) -> float:
        P = np.zeros((self._nparam, self._nparam))
        q = np.zeros(self._nparam)
        if max:
            q[idx] = -1.0
        else:
            q[idx] = 1.0

        Fd = np.array(
            self._Fd(xprev, uprev)
        ).flatten()
        Gd = np.array(
            self._Gd(xprev, uprev)
        )

        G = np.block([[Gd], [-Gd]])
        h = np.block(
            [x-Fd-d_min, -x+Fd+d_max]
        )

        param_bd = proxqp_solve_qp(
            P=P, q=q, G=G, h=h, lb=param_min, ub=param_max,
            verbose=False, backend="dense", eps_abs=10**-3, max_iter=max_iter,
        )
        try:
            return param_bd[idx]
        except TypeError:
            if not max:
                return param_min[idx]
            else:
                return param_max[idx]


    def solve_proj(
        self,
        param,
        param_min,
        param_max,
        max_iter: int
    ) -> np.ndarray:
        P = np.eye(self._nparam)
        q = np.zeros(self._nparam)
        lb = param_min-param
        ub = param_max-param
        param_err = proxqp_solve_qp(
            P=P, q=q, lb=lb, ub=ub,
            verbose=False, backend="dense", eps_abs=10**-3, max_iter=max_iter,
        )

        try:
            param_proj = param_err + param
            return param_proj
        except TypeError:
            return param
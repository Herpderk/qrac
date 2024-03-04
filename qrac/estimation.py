import casadi as cs
import numpy as np
from qpsolvers.solvers.proxqp_ import proxqp_solve_qp
from typing import Tuple
import time
from qrac.models import Quadrotor, ParameterAffineQuadrotor


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
        max_iter=10,
    ) -> None:
        self._nx = model.nx
        self._est = estimator
        model_aug = ParameterAffineQuadrotor(model)
        self._Fd, self._Gd, self._Gd_T = \
            self._get_discrete_dynamics(model_aug, self._nx, time_step)

        self._np = model_aug.np
        self._d_min = disturb_min
        self._d_max = disturb_max
        self._p_tol = param_tol
        self._sol_tol = qp_tol
        self._max_iter = max_iter

        self._p_min = param_min
        self._p_max = param_max
        self._start = False


    def _get_discrete_dynamics(
        self,
        model: ParameterAffineQuadrotor,
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
        xprev: np.ndarray,
        uprev: np.ndarray,
        param: np.ndarray,
        timer=False
    ) -> np.ndarray:
        st = time.perf_counter()
        p_min, p_max = self._sm_update(x, xprev, uprev)
        self._update_param_bds(p_min, p_max)
        p_est = self._est.get_param(x, xprev, uprev, param, False)
        p_proj = self._solve_proj(param=p_est)
        print(f"params: {p_proj}")
        if timer:
            et = time.perf_counter()
            print(f"sm runtime: {et - st}")
        return p_proj


    def _sm_update(
        self,
        x: np.ndarray,
        xprev: np.ndarray,
        uprev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._start == False:
            self._start = True
            xprev = x

        p_min = self._p_min
        p_max = self._p_max
        if not (p_max - p_min < self._p_tol).all():
            sol_min = np.zeros(self._np)
            sol_max = np.zeros(self._np)

            for i in range(self._np):
                if p_max[i] - p_min[i] > self._p_tol[i]:
                    sol_min[i] = self._solve_lp(
                        idx=i, x=x, xprev=xprev,
                        uprev=uprev, max=False
                    )
                    sol_max[i] = self._solve_lp(
                        idx=i, x=x, xprev=xprev,
                        uprev=uprev, max=True
                    )
                else:
                    sol_min[i] = p_min[i]
                    sol_max[i] = p_max[i]

            param_min = np.maximum(sol_min, p_min)
            param_max = np.minimum(sol_max, p_max)
        return param_min, param_max


    def _solve_lp(
        self,
        idx: int,
        x: np.ndarray,
        xprev: np.ndarray,
        uprev: np.ndarray,
        max: bool
    ) -> float:
        P = np.zeros((self._np, self._np))
        q = np.zeros(self._np)
        if max: q[idx] = -1.0
        else: q[idx] = 1.0

        Fd = np.array(self._Fd(xprev, uprev)).flatten()
        Gd = np.array(self._Gd(xprev, uprev))
        G = np.block([[Gd], [-Gd]])
        h = np.block([x-Fd-self._d_min, -x+Fd+self._d_max])

        p_bd = proxqp_solve_qp(
            P=P, q=q, G=G, h=h,
            lb=self._p_min, ub=self._p_max,
            verbose=False, backend="dense",
            eps_abs=self._sol_tol, max_iter=self._max_iter,
        )
        try:
            return p_bd[idx]
        except TypeError:
            if max: return self._p_max[idx]
            else: return self._p_min[idx]


    def _solve_proj(
        self,
        param: np.ndarray,
    ) -> np.ndarray:
        P = np.eye(self._np)
        q = np.zeros(self._np)
        lb = self._p_min - param
        ub = self._p_max - param
        p_err = proxqp_solve_qp(
            P=P, q=q, lb=lb, ub=ub,
            verbose=False, backend="dense",
            eps_abs=self._sol_tol, max_iter=self._max_iter,
        )
        try:
            p_proj = p_err + param
            return p_proj
        except TypeError:
            return param


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
        update_gain: float,
        time_step: float,
    ) -> None:
        self._nx = model.nx
        model_aug = ParameterAffineQuadrotor(model)
        self._Fd, self._Gd, self._Gd_T = \
            self._get_discrete_dynamics(model_aug, self._nx, time_step)
        self._mu = update_gain


    def _get_discrete_dynamics(
        self,
        model: ParameterAffineQuadrotor,
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
        xprev: np.ndarray,
        uprev: np.ndarray,
        param: np.ndarray,
        timer: bool
    ) -> np.ndarray:
        st = time.perf_counter()
        x_err = x - self._Fd(xprev, uprev) - self._Gd(xprev, uprev)@param
        p_lms = param + self._mu*self._Gd_T(xprev, uprev)@x_err
        p_lms = np.array(p_lms).flatten()
        if timer:
            et = time.perf_counter()
            print(f"LMS runtime: {et - st}")
        return p_lms


class LinearMHE():
    def __init__(self) -> None:
        pass


class NonlinearMHE():
    def __init__(self) -> None:
        pass

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
        qp_tol: float,
        max_iter: int,
    ) -> None:
        self._nx = model.nx
        self._param_est = estimator
        model_aug = ParameterAffineQuadrotor(model)
        self._Fd, self._Gd, self._Gd_T = \
            self._get_discrete_dynamics(model_aug, self._nx, time_step)

        self._nparam = model_aug.nparam
        self._d_min = disturb_min
        self._d_max = disturb_max
        self._param_tol = param_tol
        self._solver_tol = qp_tol
        self._max_iter = max_iter

        self._param_min = param_min
        self._param_max = param_max
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
        est_param = self._param_est.get_param(x, xprev, uprev, param, False)
        param_min, param_max = self._sm_update(x, xprev, uprev)
        param_proj = self._solve_proj(
            param=est_param, param_min=param_min, param_max=param_max,
            tol=self._solver_tol, max_iter=self._max_iter
        )
        self._update_param_bds(param_proj, param_min, param_max)
        print(f"params: {param_proj}")
        if timer:
            et = time.perf_counter()
            print(f"SetMembership and {type(self._param_est)} runtime: {et - st}")
        return param_proj


    def _sm_update(
        self,
        x: np.ndarray,
        xprev: np.ndarray,
        uprev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._start == False:
            self._start = True
            xprev = x

        param_min = self._param_min
        param_max = self._param_max
        if not (param_max - param_min < self._tol).all():
            sol_min = np.zeros(self._nparam)
            sol_max = np.zeros(self._nparam)

            for i in range(self._nparam):
                if param_max[i] - param_min[i] > self._tol[i]:
                    sol_min[i] = self._solve_lp(
                        idx=i, x=x, xprev=xprev, uprev=uprev,
                        d_min=self._d_min, d_max=self._d_max,
                        param_min=param_min, param_max=param_max,
                        tol=self._solver_tol, max_iter=self._max_iter,
                        max=False
                    )
                    sol_max[i] = self._solve_lp(
                        idx=i, x=x, xprev=xprev, uprev=uprev,
                        d_min=self._d_min, d_max=self._d_max,
                        param_min=param_min, param_max=param_max,
                        tol=self._solver_tol, max_iter=self._max_iter,
                        max=True
                    )
                else:
                    sol_min[i] = param_min[i]
                    sol_max[i] = param_max[i]

            param_min = np.maximum(sol_min, param_min)
            param_max = np.minimum(sol_max, param_max)
        return param_min, param_max


    def _solve_lp(
        self,
        idx: int,
        x: np.ndarray,
        xprev: np.ndarray,
        uprev: np.ndarray,
        d_min: np.ndarray,
        d_max: np.ndarray,
        param_min: np.ndarray,
        param_max: np.ndarray,
        tol: float,
        max_iter: int,
        max: bool
    ) -> float:
        P = np.zeros((self._nparam, self._nparam))
        q = np.zeros(self._nparam)
        if max: q[idx] = -1.0
        else: q[idx] = 1.0

        Fd = np.array(self._Fd(xprev, uprev)).flatten()
        Gd = np.array(self._Gd(xprev, uprev))
        G = np.block([[Gd], [-Gd]])
        h = np.block([x-Fd-d_min, -x+Fd+d_max])

        param_bd = proxqp_solve_qp(
            P=P, q=q, G=G, h=h, lb=param_min, ub=param_max,
            verbose=False, backend="dense",
            eps_abs=tol, max_iter=max_iter,
        )
        try:
            return param_bd[idx]
        except TypeError:
            if max: return param_max[idx]
            else: return param_min[idx]


    def _solve_proj(
        self,
        param: np.ndarray,
        param_min: np.ndarray,
        param_max: np.ndarray,
        tol: float,
        max_iter: int
    ) -> np.ndarray:
        P = np.eye(self._nparam)
        q = np.zeros(self._nparam)
        lb = param_min-param
        ub = param_max-param
        param_err = proxqp_solve_qp(
            P=P, q=q, lb=lb, ub=ub,
            verbose=False, backend="dense",
            eps_abs=tol, max_iter=max_iter,
        )
        try:
            param_proj = param_err + param
            return param_proj
        except TypeError:
            return param


    def _update_param_bds(
        self,
        param_min: np.ndarray,
        param_max: np.ndarray,
    ) -> None:
        self._param_min = param_min
        self._param_max = param_max


class LeastMeanSquareEstimator():
    def __init__(
        self,
        model: Quadrotor,
        update_gain: float,
        time_step: float,
    ) -> None:
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
        param_lms = param + self._mu*self._Gd_T(xprev, uprev)@x_err
        param_lms = np.array(param_lms).flatten()
        if timer:
            et = time.perf_counter()
            print(f"LMS runtime: {et - st}")
        return param_lms


class LinearMHE():
    def __init__(self) -> None:
        pass


class NonlinearMHE():
    def __init__(self) -> None:
        pass

#!/usr/bin/python3

import casadi as cs
import numpy as np
from scipy.linalg import block_diag
import os
import contextlib
import sys
from typing import Tuple
import time
from qrac.dynamics import Quadrotor, ParameterAffineQuadrotor
from qrac.control.nmpc import NMPC

'''
@contextlib.contextmanager
def suppress_print():
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        yield


class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout
'''

class SetMembershipMPC():
    def __init__(
        self,
        model: Quadrotor,
        Q: np.ndarray,
        R: np.ndarray,
        update_gain: float,
        param_max: np.ndarray,
        param_min: np.ndarray,
        disturb_min: np.ndarray,
        disturb_max: np.ndarray,
        u_max: np.ndarray,
        u_min: np.ndarray,
        time_step: float,
        num_nodes: int,
        #moving_window_step: float,
        #moving_window_len: int,
        real_time: bool,
    ) -> None:
        self._nx = model.nx
        self._nu = model.nu
        self._dt = time_step
        self._N = num_nodes
        self._rt = real_time
        #self._loop_ratio = int(round(moving_window_step / time_step))
        #self._loop_ct = int(round(moving_window_step / time_step))

        self._nparam = 10
        self._param_idx = 12
        self._d_min = disturb_min
        self._d_max = disturb_max
        self._mu = update_gain
        #self._dtM = moving_window_step
        #self._M = moving_window_len

        model_aug = ParameterAffineQuadrotor(model)
        self._Fd_func, self._Gd_func, self._Gd_T_func = \
            self._get_dynamics_funcs(model_aug)
        self._min_lps, self._max_lps = self._init_lps(model_aug)
        self._proj = self._init_proj()

        Q_aug = self._augment_costs(Q)
        self._mpc = NMPC(
            model_aug, Q_aug, R, u_min, u_max,
            time_step, num_nodes, real_time
        )

        self._param_min = param_min
        self._param_max = param_max
        self._param = model_aug.get_parameters()
        self._x = np.zeros(self._nx)
        self._u = np.zeros(self._nu)


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
        if timer: st = time.perf_counter()

        param = self._update_params(x)

        x_aug = np.concatenate((x, param))
        xset_aug = self._augment_xset(xset)
        self._x = x[:self._nx]
        self._u = self._mpc.get_input(x_aug, xset_aug, False)
        if timer:
            print(f"mpc runtime: {time.perf_counter() - st}")
        return self._u


    def get_state(
        self,
        x: np.ndarray,
        xset: np.ndarray,
        timer=False,
    ) -> np.ndarray:
        if timer: st = time.perf_counter()
        param = self._update_params(x)
        x_aug = np.concatenate((x, param))
        xset_aug = self._augment_xset(xset)
        x1 = self._mpc.get_state(x_aug, xset_aug, False)
        if timer:
            print(f"mpc runtime: {time.perf_counter() - st}")
        return x1


    def get_trajectory(
        self,
        x: np.ndarray,
        xset: np.ndarray,
        timer=False,
        visuals=False
    ) -> Tuple[np.ndarray]:
        if timer: st = time.perf_counter()
        #self._timer.value = timer
        #self._update_loop_ct()
        param = self._update_params(x)
        x_aug = np.concatenate((x, param))
        xset_aug = self._augment_xset(xset)
        xs, us = self._mpc.get_trajectory(x_aug, xset_aug, False, visuals)
        if timer:
            print(f"mpc runtime: {time.perf_counter() - st}")
        return xs, us


    def _update_params(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        param = self._lms_update(x)
        param_min, param_max = self._sm_update(x)
        param = self._solve_proj(param, param_min, param_max)
        self._update_vars(param, param_min, param_max)
        return param


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


    def _update_loop_ct(self) -> None:
        if self._loop_ct == self._loop_ratio:
            self._loop_ct = 0
        self._loop_ct += 1


    def _sm_update(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        min_sols = np.zeros(self._nparam)
        max_sols = np.zeros(self._nparam)
        for i in range(self._nparam):
            if not self._param_min[i] == self._param_max[i]:
                min_sols[i], max_sols[i] = self._solve_lps(i, x,)

        sol_min = np.minimum(
            min_sols, max_sols
        )
        sol_max = np.maximum(
            min_sols, max_sols
        )
        param_min = np.maximum(
            sol_min, self._param_min
        )
        param_max = np.minimum(
            sol_max, self._param_max
        )
        return param_min, param_max


    def _solve_lps(
            self,
            idx,
            x,
        ):
        #with suppress_print():
        p_min = np.concatenate(
            (self._param_min, x, self._x, self._u)
        )
        min_lpsol = self._min_lps[idx](
            x0=self._param, p=p_min,
            lbx=self._param_min, ubx=self._param_max
        )
        min_sol = np.array(min_lpsol["x"][idx]).flatten()

        p_max = np.concatenate(
            (self._param_max, x, self._x, self._u)
        )
        max_lpsol = self._max_lps[idx](
            x0=self._param, p=p_max,
            lbx=self._param_min, ubx=self._param_max
        )
        max_sol = np.array(max_lpsol["x"][idx]).flatten()
        return min_sol, max_sol


    def _lms_update(
        self,
        x,
    ) -> np.ndarray:
        x_err = x - self._Fd_func(self._x, self._u) - \
            self._Gd_func(self._x, self._u)@self._param
        param_lms = self._param + \
            self._mu*self._Gd_T_func(self._x, self._u)@x_err
        param_lms = np.array(param_lms).flatten()
        return param_lms


    def _solve_proj(
        self,
        param,
        param_min,
        param_max
    ):
        if not (param_min == param_max).all():
            #with suppress_print():
                projsol = self._proj(
                    x0=param, p=param,
                    lbx=param_min, ubx=param_max
                )
                param_proj = np.array(projsol["x"]).flatten()
        return param_proj


    def _update_vars(
        self,
        param,
        param_min,
        param_max
    ) -> None:
        self._param = param
        self._param_min = param_min
        self._param_max = param_max
        print(f"param_min: {param_min}")
        print(f"param_max: {param_max}")
        print(f"params: {param}")


    def _init_proj(self) -> cs.nlpsol:
        param = cs.SX.sym("param_lms", self._nparam)
        param_proj = cs.SX.sym("param_proj", self._nparam)
        projprob = {
            "p": param,
            "x": param_proj,
            "f": cs.norm_1((param_proj - param))
        }
        opts = {
            "verbose": False
        }
        proj = cs.qpsol("proj", "osqp", projprob, opts)
        return proj


    def _init_lps(
        self,
        model: ParameterAffineQuadrotor
    ) -> cs.qpsol:
        nparam = self._nparam
        nx = self._nx

        param_bd = cs.SX.sym("param_bd", self._nparam)
        x_curr = cs.SX.sym("x_curr", self._nx)
        x_prev = model.x[:self._nx]
        u_prev = model.u
        param = model.x[self._param_idx:self._param_idx+self._nparam]
        min_lps = []
        max_lps = []

        for i in range(self._nparam):
            e = cs.SX.zeros(self._nparam)
            e[i] = 1.0
            min_lp = {
                "p": cs.vertcat(param_bd, x_curr, x_prev, u_prev,),
                "x": param,
                "f": e.T@param,
                "g": x_curr - self._dt*(model.F[:nx] - model.G[:nx,:nparam]@param)
            }
            max_lp = {
                "p": cs.vertcat(param_bd, x_curr, x_prev, u_prev),
                "x": param,
                "f": -e.T@param,
                "g": x_curr - self._dt*(model.F[:nx] - model.G[nx,:nparam]@param)
            }
            opts = {
                "verbose": False
            }
            min_lp = cs.qpsol(f"min_lpsol_{i}", "clp", min_lp, opts)
            max_lp = cs.qpsol(f"max_lpsol_{i}", "clp", max_lp, opts)

            min_lps += [min_lp]
            max_lps += [max_lp]
        return min_lps, max_lps


    def _augment_costs(
        self,
        Q: np.ndarray,
    ) -> np.ndarray:
        Q_aug = block_diag(Q, np.zeros((self._nparam, self._nparam)))
        return Q_aug


    def _get_dynamics_funcs(
        self,
        model: ParameterAffineQuadrotor,
    ) -> Tuple[cs.Function]:
        Fd = model.x[:self._nx] + self._dt * model.F[:self._nx]
        Gd = self._dt * model.G[:self._nx,:]
        Gd_T = cs.transpose(Gd)

        Fd_func = cs.Function(
            "Fd",
            [model.x[:self._nx], model.u],
            [Fd[:self._nx]]
        )
        Gd_func = cs.Function(
            "Gd",
            [model.x[:self._nx], model.u],
            [Gd[:self._nx, :self._nparam]]
        )   
        Gd_T_func = cs.Function(
            "Gd_T",
            [model.x[:self._nx], model.u],
            [Gd_T[:self._nparam, :self._nx]]
        )
        return Fd_func, Gd_func, Gd_T_func


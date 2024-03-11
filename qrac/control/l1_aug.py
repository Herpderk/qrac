#!/usr/bin/python3

import casadi as cs
import numpy as np
import multiprocessing as mp
from scipy.linalg import expm
from typing import Tuple
import time
from qrac.models import Quadrotor


class L1Augmentation():
    def __init__(
        self,
        model: Quadrotor,
        control_ref,
        adapt_gain: float,
        bandwidth: float,
    ) -> None:
        self._z_idx = 6
        self._nz = 6
        self._nm = 4
        self._num = 2
        self._assert(model, control_ref, adapt_gain, bandwidth,)

        self._u_l1 = np.zeros(model.nu)
        self._z = np.zeros(self._nz)
        self._dstb_m = np.zeros(self._nm)
        self._dstb_um = np.zeros(self._num)
        self._f, self._g_m, self._g_um = self._get_dynamics_funcs(model)
        self._Am = -np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        self._adapt_gain = adapt_gain
        self._adapt_exp, self._adapt_mat, self._filter_exp = \
            self._get_l1_const(bandwidth, control_ref.dt)

        self._model = model
        self._ctrl_ref = control_ref
        self._dt = control_ref.dt

        self._x = np.zeros(model.nx)
        self._xset = np.zeros(control_ref.n_set)
        self._u_ref = np.zeros(model.nu)


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
        timer=False,
    ) -> np.ndarray:
        assert x.shape[0] == self._model.nx
        assert xset.shape[0] == self._ctrl_ref.n_set
        uref = self._ctrl_ref.get_input(x=x, xset=xset, timer=timer)
        ul1 = self._get_l1_input(x=x, uref=uref, timer=timer)
        u = uref + ul1
        return u


    def _get_l1_input(
        self,
        x: np.ndarray,
        uref: np.ndarray,
        timer: bool
    ) -> None:
        st = time.perf_counter()
        z_err, g_m, g_um = self._predictor(x=x, uref=uref)
        dstb_m = self._adaptation(z_err=z_err, g_m=g_m, g_um=g_um)
        ul1 = self._control_law(dstb_m=dstb_m)
        if timer:
            print(f"L1 runtime: {time.perf_counter() - st}")
        return ul1


    def _predictor(
        self,
        x: np.ndarray,
        uref: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z_err = np.array(
            self._z - x[self._z_idx : self._z_idx+self._nz]
        )
        f = np.array(self._f(x, uref)).reshape(self._nz)

        g_m = np.array(self._g_m(x))
        g_um = np.array(self._g_um(x))
        ul1 = self._u_l1
        d_m = self._dstb_m
        d_um = self._dstb_um
        self._z += self._dt * (self._Am@z_err + \
            f + g_m@(ul1+d_m) + g_um@d_um)
        return z_err, g_m, g_um


    def _adaptation(
        self,
        z_err,
        g_m: np.ndarray,
        g_um: np.ndarray
    ) -> np.ndarray:
        I = np.eye(self._nz)
        G = np.block([g_m, g_um])
        mu = self._adapt_exp @ z_err
        adapt = self._adapt_gain * -I @ np.linalg.inv(G) @ self._adapt_mat @ mu
        self._dstb_m = adapt[: self._nm]
        self._dstb_um = adapt[self._nm : self._nm + self._num]
        return self._dstb_m


    def _control_law(
        self,
        dstb_m: np.ndarray
    ) -> None:
        self._u_l1 = self._filter_exp*self._u_l1 - \
            (1-self._filter_exp)*dstb_m
        return self._u_l1


    def _get_l1_const(
        self,
        bandwidth: float,
        time_step: float
    ) -> Tuple[np.ndarray]:
        adapt_exp = expm(self._Am*time_step)
        adapt_mat = np.linalg.inv(self._Am) @ (adapt_exp - np.eye(self._nz))
        filter_exp = np.exp(-bandwidth*time_step)
        return adapt_exp, adapt_mat, filter_exp


    def _get_dynamics_funcs(
        self,
        model: Quadrotor,
    ) -> Tuple[cs.Function]:
        # rotation matrix from body frame to inertial frame
        b1 = model.R[:,0]
        b2 = model.R[:,1]
        b3 = model.R[:,2]

        f_sym = model.xdot[6:12]
        g_m_sym = cs.SX(cs.vertcat(
            b3/model.m @ cs.SX.ones(1,4),
            cs.inv(model.J) @ model.B
        ))
        g_um_sym = cs.SX(cs.vertcat(
            cs.horzcat(b1, b2)/model.m,
            cs.SX.zeros(3,2)
        ))

        f = cs.Function("f", [model.x, model.u], [f_sym])
        g_m = cs.Function("g_m", [model.x], [g_m_sym])
        g_um = cs.Function("g_um", [model.x], [g_um_sym])
        return f, g_m, g_um


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
                "Please input the cutoff frequency as an integer or float!")

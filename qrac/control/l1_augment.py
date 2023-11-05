#!/usr/bin/python3

import casadi as cs
import numpy as np
import multiprocessing as mp
from typing import Tuple
import time
from qrac.dynamics import NonlinearQuadrotor


class L1Augmentation():
    def __init__(
        self,
        model: NonlinearQuadrotor,
        control_ref,
        adapt_gain: np.ndarray,
        cutoff_freq: float,
        time_step: float,
        real_time: bool,
    ) -> None:
        assert control_ref.dt >= time_step
        self._assert()
        self._nx, self._nu = self._get_dims(model)
        self._ctrl_ref = control_ref
        self._dt = time_step
        self._rt = real_time
        self._loop_ratio = int(round(control_ref.dt / time_step))
        self._loop_ct = int(round(control_ref.dt / time_step))

        self._x = mp.Array("f", np.zeros(self._nx))
        self._x_set = mp.Array("f", np.zeros(self._nx))
        self._u_ref = mp.Array("f", np.zeros(self._nu))
        self._timer = mp.Value("b", False)

        self._u_l1 = np.zeros(self._nu)
        self._dstb_m = np.zeros(4)
        self._dstb_um = np.zeros(2)
        self._z = np.zeros(6)
        self._f, self._g_m, self._g_um = self._get_l1_dynamics(model)
        self._As = adapt_gain
        self._adapt_exp, self._adapt_mat, self._filter_exp = \
            self._get_l1_const(adapt_gain, cutoff_freq, time_step)

        if real_time:
            self._run_flag = mp.Value("b", False)


    @property
    def dt(self) -> float:
        return self._dt


    def get_input(
        self,
        x0: np.ndarray,
        x_set: np.ndarray,
        timer=False,
    ) -> np.ndarray:
        self._timer.value = timer
        self._x[:] = x0
        self._x_set[:] = x_set
        if self._rt:
            self._update_rt()
        else:
            self._update_non_rt()
        self._run_l1()
        u = np.array(self._u_ref[:]) + self._u_l1
        return u


    def _update_rt(self) -> None:
        if self._run_flag.value:
            self._run_flag.value = True


    def _update_non_rt(self) -> None:
        if self._loop_ct == self._loop_ratio:
            self._loop_ct = 0
            self._update_ctrl_ref()
        self._loop_ct += 1


    def _run_ctrl_ref(self) -> None:
        while True:
            if self._run_flag:
                self._update_ctrl_ref()


    def _update_ctrl_ref(self) -> np.ndarray:
        x0 = np.array(self._x[:])
        x_set = np.array(self._x_set[:])
        timer = self._timer.value
        self._u_ref = self._ctrl_ref.get_input(x0=x0, x_set=x_set, timer=timer)


    def _run_l1(self) -> None:
        st = time.perf_counter()
        self._run_predictor()
        self._run_adapt_law()
        self._run_ctrl_law()
        if self._timer.value:
            print(f"L1 runtime: {time.perf_counter() - st}")
        if self._rt:
            while time.perf_counter() - st < self._dt:
                pass


    def _run_predictor(self) -> None:
        x = np.array(self._x[:])
        u_ref = np.array(self._u_ref[:])
        z = np.array(self._z[:])
        z_err = np.array(z - x[6:12])
        f = np.array(self._f(x, u_ref)).reshape(6)
        g_m = np.array(self._g_m(x))
        g_um = np.array(self._g_um(x))
        u_l1 = self._u_l1
        dstb_m = self._dstb_m
        dstb_um = self._dstb_um
        z += + self._dt * \
            (f + g_m@(u_l1+dstb_m) + g_um@dstb_um + self._As@z_err)
        self._z[:] = z


    def _run_adapt_law(self) -> None:
        x = np.array(self._x[:])
        z = np.array(self._z[:])
        z_err = np.array(z - x[6:12])
        g_m = np.array(self._g_m(x))
        g_um = np.array(self._g_um(x))

        I = np.eye(6)
        G = np.block([g_m, g_um])
        mu = self._adapt_exp @ z_err
        adapt = -I @ np.linalg.inv(G) @ self._adapt_mat @ mu
        self._dstb_m = adapt[0:4]
        self._dstb_um = adapt[4:6]


    def _run_ctrl_law(self) -> None:
        self._u_l1 = self._filter_exp*self._u_l1 - \
            (1-self._filter_exp)*self._dstb_m


    def _get_l1_const(
        self,
        adapt_gain: np.ndarray,
        cutoff_freq: float,
        time_step: float
    ) -> Tuple[np.ndarray]:
        adapt_exp = np.exp(adapt_gain*time_step)
        adapt_mat = np.linalg.inv(adapt_gain) @ \
            (adapt_exp - np.eye(adapt_gain.shape[0]))
        filter_exp = np.exp(-cutoff_freq*time_step)
        return adapt_exp, adapt_mat, filter_exp


    def _get_l1_dynamics(
        self,
        model: NonlinearQuadrotor,
    ) -> Tuple[cs.Function]:
        phi = model.x[3]
        theta = model.x[4]
        psi = model.x[5]

        # rotation matrix from body frame to inertial frame
        Rx = cs.SX(cs.vertcat(
            cs.horzcat(1,            0,            0),
            cs.horzcat(0,  cs.cos(phi), -cs.sin(phi)),
            cs.horzcat(0,  cs.sin(phi),  cs.cos(phi)),
        ))
        Ry = cs.SX(cs.vertcat(
            cs.horzcat( cs.cos(theta),  0,  cs.sin(theta)),
            cs.horzcat(             0,  1,              0),
            cs.horzcat(-cs.sin(theta),  0,  cs.cos(theta)),
        ))
        Rz = cs.SX(cs.vertcat(
            cs.horzcat(cs.cos(psi),    -cs.sin(psi),    0),
            cs.horzcat(cs.sin(psi),     cs.cos(psi),    0),
            cs.horzcat(          0,               0,    1),
        ))
        R = Rz @ Ry @ Rx
        rx, ry, rz = cs.horzsplit(R)

        J = cs.SX(np.diag([model.Ixx, model.Iyy, model.Izz]))
        P = cs.SX(cs.vertcat(
            model.by.reshape(1, model.by.shape[0]),
            -model.bx.reshape(1, model.bx.shape[0]),
            cs.horzcat(-model.k[0], model.k[1], -model.k[2], model.k[3]),
        ))

        f_sym = model.f_expl_expr[6:12]
        g_m_sym = cs.SX(cs.vertcat(
            R/model.m @ cs.vertcat(cs.SX.zeros(2,4), cs.SX.ones(1,4)),
            cs.inv(J) @ P
        ))
        g_um_sym = cs.SX(cs.vertcat(
            R/model.m @ cs.vertcat(cs.SX.eye(2), cs.SX.zeros(1,2)),
            cs.SX.zeros(3,2)
        ))
        '''
        g_um_sym = cs.SX(cs.vertcat(
            (1/model.m) * cs.horzcat(rx, ry),
            cs.SX.zeros(3, 2)
        ))'''

        f = cs.Function("f", [model.x, model.u], [f_sym])
        g_m = cs.Function("g_m", [model.x], [g_m_sym])
        g_um = cs.Function("g_um", [model.x], [g_um_sym])
        return f, g_m, g_um


    def _get_dims(
        self,
        model: NonlinearQuadrotor,
    ) -> Tuple[int]:
        nx = model.x.shape[0]
        nu = model.u.shape[0]
        return nx, nu


    def _assert(self) -> None:
        pass

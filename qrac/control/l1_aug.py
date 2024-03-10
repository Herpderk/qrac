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
        time_step: float,
        real_time: bool,
    ) -> None:
        self._z_idx = 6
        self._nz = 6
        self._nm = 4
        self._num = 2
        self._assert(model, control_ref, adapt_gain, bandwidth, time_step, real_time)

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
            self._get_l1_const(bandwidth, time_step)

        self._model = model
        self._ctrl_ref = control_ref
        self._dt = time_step
        self._rt = real_time
        self._loop_ratio = int(round(control_ref.dt / time_step))
        self._loop_ct = int(round(control_ref.dt / time_step))

        self._x = mp.Array("f", np.zeros(model.nx))
        self._xset = mp.Array("f", np.zeros(control_ref.n_set))
        self._u_ref = mp.Array("f", np.zeros(model.nu))
        self._timer = mp.Value("b", False)
        if real_time:
            self._run_flag = mp.Value("b", True)


    @property
    def dt(self) -> float:
        return self._dt


    @property
    def n_set(self) -> Tuple[int]:
        return self._ctrl_ref.n_set


    def start(self) -> None:
        if not self._rt:
            print("This controller is not in real-time mode!")
        else:
            proc = mp.Process(target=self._ctrl_ref_proc, args=[])
            proc.start()
            

    def stop(self) -> None:
        if not self._rt:
            print("This controller is not in real-time mode!")
        else:
            self._run_flag.value = False


    def get_input(
        self,
        x: np.ndarray,
        xset: np.ndarray,
        timer=False,
    ) -> np.ndarray:
        assert x.shape[0] == self._model.nx
        assert xset.shape[0] == self._ctrl_ref.n_set
        self._timer.value = timer
        self._x[:] = x
        self._xset[:] = xset
        if not self._rt:
            self._update_non_rt()
        self._run_l1()
        u = np.array(self._u_ref[:]) + self._u_l1
        return u


    def _ctrl_ref_proc(self) -> None:
        ctrl_ref_proc = mp.Process(target=self._run_ctrl_ref, args=[])
        ctrl_ref_proc.start()
        ctrl_ref_proc.join()
        print("\nController successfully stopped.")


    def _run_ctrl_ref(self) -> None:
        st = time.perf_counter()
        while self._run_flag.value:
            et = time.perf_counter()
            if et - st >= self._ctrl_ref.dt:
                st = et
                self._update_ctrl_ref()


    def _update_non_rt(self) -> None:
        if self._loop_ct == self._loop_ratio:
            self._loop_ct = 0
            self._update_ctrl_ref()
        self._loop_ct += 1


    def _update_ctrl_ref(self) -> np.ndarray:
        x = np.array(self._x[:])
        xset = np.array(self._xset[:])
        timer = self._timer.value
        self._u_ref[:] = self._ctrl_ref.get_input(
            x=x, xset=xset, timer=timer
        )


    def _run_l1(self) -> None:
        st = time.perf_counter()
        self._predictor()
        self._adaptation()
        self._control_law()
        if self._timer.value:
            print(f"L1 runtime: {time.perf_counter() - st}")
        if self._rt:
            while time.perf_counter() - st < self._dt:
                pass


    def _predictor(self) -> None:
        x = np.array(self._x[:])
        u_ref = np.array(self._u_ref[:])
        z = np.array(self._z[:])
        z_err = np.array(z - x[self._z_idx : self._z_idx+self._nz])
        f = np.array(self._f(x, u_ref)).reshape(self._nz)
        g_m = np.array(self._g_m(x))
        g_um = np.array(self._g_um(x))
        u_l1 = self._u_l1
        dstb_m = self._dstb_m
        dstb_um = self._dstb_um
        z += + self._dt * \
            (f + g_m@(u_l1+dstb_m) + g_um@dstb_um + self._Am@z_err)
        self._z[:] = z


    def _adaptation(self) -> None:
        x = np.array(self._x[:])
        z = np.array(self._z[:])
        z_err = np.array(z - x[self._z_idx : self._z_idx+self._nz])
        g_m = np.array(self._g_m(x))
        g_um = np.array(self._g_um(x))

        I = np.eye(self._nz)
        G = np.block([g_m, g_um])
        mu = self._adapt_exp @ z_err
        adapt = self._adapt_gain * -I @ np.linalg.inv(G) @ self._adapt_mat @ mu
        self._dstb_m = adapt[: self._nm]
        self._dstb_um = adapt[self._nm : self._nm + self._num]


    def _control_law(self) -> None:
        self._u_l1 = self._filter_exp*self._u_l1 - \
            (1-self._filter_exp)*self._dstb_m


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
        time_step: float,
        real_time: bool,
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
        if type(time_step) != int and type(time_step) != float:
            raise ValueError(
                "Please input the control loop step as an integer or float!")
        if type(real_time) != bool:
            raise ValueError(
                "Please input the real-time mode as a bool!")
        if control_ref.dt < time_step:
            raise ValueError(
                "Please make sure the reference controller time step is greater than or equal to the L1 time step!")

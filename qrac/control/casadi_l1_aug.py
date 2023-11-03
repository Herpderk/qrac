#!/usr/bin/python3

import casadi as cs
import numpy as np
import multiprocessing as mp
from typing import Tuple
from qrac.dynamics.casadi import CasadiModel


class CasadiL1Augmentation():
    def __init__(
        self,
        control_ref,
        model_ref,
        adapt_gain: np.ndarray,
        cutoff_freq: float,
        time_step: float,
        real_time: bool,
    ) -> None:
        assert type(model_ref) == CasadiModel
        assert control_ref.dt >= time_step

        self._nx, self._nu = self._get_dims(model_ref)
        self._ctrl_ref = control_ref
        self._model_ref = model_ref
        self._dt = time_step
        self._rt = real_time

        self._x = mp.Array("f", np.zeros(self._nx))
        self._x_set = mp.Array("f", np.zeros(self._nx))
        self._timer = mp.Value("b", False)
        self._u_l1 = np.zeros(self._nu)
        self._u_ref = mp.Array("f", np.zeros(self._nu))
        
        self._adapt_mat, self._filter_mat. self._z = \
            self._init_l1_vars(adapt_gain, cutoff_freq, time_step)

        if real_time:
            self._run_flag = mp.Value("b", False)
        self._u_l1 = np.zeros(self._nu)
        self._u_ref = mp.Array("f", np.zeros(self._nu))


    @property
    def dt(self) -> float:
        return self._dt


    def get_input(
        self,
        x0: np.ndarray,
        x_set: np.ndarray,
        timer=False,
    ) -> np.ndarray:
        if self._rt:    # share info with outer ref controller if running real-time mode
            self._update_rt(x0, x_set, timer)
        else:           # or run it in sync in non-real time
            self._update_non_rt()
        self._update_l1()   # run L1 loop
        u = np.array(self._u_ref[:]) + self._u_l1
        return u


    def _update_rt(
        self,
        x0: np.ndarray,
        x_set: np.ndarray,
        timer: bool,
    ) -> None:
        self._timer.value = timer
        self._x[:] = x0
        self._x_set[:] = x_set
        if self._run_flag.value:
            self._run_flag.value = True


    def _update_non_rt(self) -> None:
        self._loop_ct += 1
        if self._loop_ct == self._loop_ratio:
            self._loop_ct = 0
            self._u_ref[:] = self._get_ctrl_ref()


    def _run_ctrl_ref(self) -> None:
        while True:
            if self._run_flag:
                self._u_ref[:] = self._get_ctrl_ref()


    def _get_ctrl_ref(self) -> np.ndarray:
        x0 = np.array(self._x[:])
        x_set = np.array(self._x_set[:])
        timer = self._timer.value
        u_ref = self._ctrl_ref.get_input(x0=x0, x_set=x_set, timer=timer)
        return u_ref


    def _update_l1(self) -> None:
        self._run_predictor()
        self._run_adapt_law()
        self._run_control_law()


    def _run_predictor(self):
        pass


    def _run_adapt_law(self):
        pass


    def _run_control_law(self):
        self._u_l1 += 0 # f, g, disturbance, prediction err stuff * dt


    def _init_l1_vars(
        self,
        adapt_gain: np.ndarray,
        cutoff_freq: float,
        time_step: float
    ) -> Tuple:
        adapt_mat = np.linalg.inv(adapt_gain) @ \
            (np.exp(adapt_gain*time_step) - np.eye(adapt_gain.shape[0]))
        filter_mat = np.exp(-cutoff_freq*time_step)
        z = np.zeros(3)
        return adapt_mat, filter_mat, z


    def _get_dims(self, model):
        nx = model.x.shape[0]
        nu = model.u.shape[0]
        return nx, nu
        

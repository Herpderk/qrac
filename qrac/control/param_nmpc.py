#!/usr/bin/python3

import numpy as np
import multiprocessing as mp
from scipy.linalg import block_diag
from qrac.models import Quadrotor, ParameterAffineQuadrotor
from qrac.control.nmpc import NMPC


def npify(arr_like) -> np.ndarray:
    return np.array(arr_like[:])


class ParameterAdaptiveNMPC():
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
        rti: bool,
        real_time: bool,
        nlp_tol=10**-6,
        nlp_max_iter=10,
        qp_max_iter=10,
    ) -> None:
        self._nx = model.nx
        self._nu = model.nu
        self._N = num_nodes
        self._rt = real_time

        model_aug = ParameterAffineQuadrotor(model)
        self._np = model_aug.np
        Q_aug = self._augment_costs(Q)
        self._mpc = NMPC(
            model=model_aug, Q=Q_aug, R=R,
            u_min=u_min, u_max=u_max,
            time_step=time_step,
            num_nodes=num_nodes, rti=rti,
            nlp_tol=nlp_tol, nlp_max_iter=nlp_max_iter,
            qp_max_iter=qp_max_iter
        )
        self._est = estimator

        self._p = mp.Array("f", model_aug.get_parameters())
        self._x = mp.Array("f", np.zeros(model.nx))
        self._xprev = mp.Array("f", np.zeros(self._nx))
        self._uprev = mp.Array("f", np.zeros(self._nu))
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
            print("This controller is not in real-time mode!")
        else:
            proc = mp.Process(target=self._param_proc, args=[])
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
        self._xprev[:] = self._x[:]
        self._x[:] = x
        self._timer.value = timer
        if not self._rt: self._get_param()

        x_aug = np.concatenate((x, npify(self._p)))
        xset_aug = self._augment_xset(xset)
        self._uprev[:] = self._mpc.get_input(x_aug, xset_aug, timer)
        return npify(self._uprev)


    def _augment_xset(
        self,
        xset: np.ndarray
    ) -> np.ndarray:
        nx = self._nx
        xset_aug = np.zeros(self._N * (nx+self._np))
        for k in range(self._N):
            xset_aug[k*(nx+self._np) : k*(nx+self._np) + nx] =\
                xset[k*nx : k*nx + nx]
        return xset_aug


    def _param_proc(self) -> None:
        param_proc = mp.Process(target=self._run_param_est)
        param_proc.start()
        param_proc.join()
        print("\nParameter Estimator successfully stopped.")


    def _run_param_est(self) -> None:
        while self._run_flag.value:
            self._get_param()


    def _get_param(self) -> None:
        param = self._est.get_param(
            x=npify(self._x),
            xprev=npify(self._xprev),
            uprev=npify(self._uprev),
            param=npify(self._p),
            timer=self._timer.value
        )
        self._p[:] = param


    def _augment_costs(
        self,
        Q: np.ndarray,
    ) -> np.ndarray:
        Q_aug = block_diag(
            Q, np.zeros((self._np, self._np))
        )
        return Q_aug

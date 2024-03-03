#!/usr/bin/python3

from acados_template import AcadosSim, AcadosSimSolver
import numpy as np
import time
import atexit
import os
from qrac.models import Quadrotor, DisturbedQuadrotor


class AcadosPlant():
    def __init__(
        self,
        model: Quadrotor,
        sim_step: float,
        control_step: float,
    ) -> None:
        self._assert(model, sim_step, control_step)
        self._model = DisturbedQuadrotor(model)
        self._nx = model.nx
        self._nu = model.nu
        self._nd = self._model.nd
        self._solver = self._init_solver(self._model, sim_step, control_step)
        atexit.register(self._clear_files)


    @property
    def nx(self) -> int:
        return self._nx


    @property
    def nu(self) -> int:
        return self._nu


    @property
    def nd(self) -> int:
        return self._nd


    def update(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        timer=False,
    ) -> np.ndarray:
        u = np.where(u>self._model.u_min, u, self._model.u_min)
        u = np.where(u<self._model.u_max, u, self._model.u_max)
        self._solve(x, u, d, timer)
        x = self._solver.get("x")[:self._nx]
        if x[2] < 0:
            x[2] = 0
            if x[8] < 0:
                x[8] = 0
        return x


    def _solve(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        timer: bool,
    ) -> None:
        st = time.perf_counter()
        assert len(x) == self._nx
        assert len(u) == self._nu
        assert len(d) == self._nd
        self._solver.set("x", np.concatenate((x,d)))
        self._solver.set("u", u)
        status = self._solver.solve()
        if status != 0:
            raise Exception(f"acados returned status {status}.")
        if timer:
            print(f"plant runtime: {time.perf_counter() - st}")


    def _init_solver(
        self,
        model: Quadrotor,
        sim_step: float,
        control_step: float,
    ) -> AcadosSimSolver:
        sim = AcadosSim()
        sim.model = model.get_acados_model()
        sim.solver_options.T = control_step
        sim.solver_options.integrator_type = "ERK"
        sim.solver_options.num_stages = 4
        sim.solver_options.num_steps = int(round(control_step / sim_step))
        solver = AcadosSimSolver(sim)
        return solver


    def _assert(
        self,
        model: Quadrotor,
        sim_step: float,
        control_step: float,
    ) -> None:
        if type(model) != Quadrotor:
            raise TypeError(
                "The inputted model must be of type 'AcadosModel'!")
        if (type(sim_step) != int and type(sim_step) != float):
            raise TypeError(
                "Please input the desired simulator step as an integer or float!")
        if (type(control_step) != int and type(control_step) != float):
            raise TypeError(
                "Please input the desired control loop step as an integer or float!")
        if control_step < sim_step:
            raise ValueError(
                "The control step should be greater than or equal to the simulator step!")        


    def _clear_files(self) -> None:
        """
        Clean up the acados generated files.
        """
        try: os.remove("acados_sim.json")
        except: print("acados_sim.json")

#!/usr/bin/python3

from acados_template import AcadosSim, AcadosSimSolver, AcadosModel
import numpy as np
import time
import atexit
import os
from typing import Tuple


class AcadosBackend():
    def __init__(
        self,
        model,
        sim_step,
        control_step,
    ) -> None:
        assert type(model) == AcadosModel
        assert (type(sim_step) == int or type(sim_step) == float)
        assert (type(control_step) == int or type(control_step) == float)
        assert control_step > sim_step

        self._nx, self._nu = self._get_dims(model)
        self._solver = self._init_solver(model, sim_step, control_step)
        atexit.register(self._clear_files)


    @property
    def nx(self) -> int:
        return self._nx


    @property
    def nu(self) -> int:
        return self._nu


    def update(
        self,
        x0: np.ndarray,
        u: np.ndarray,
        timer=False,
    ) -> np.ndarray:
        self._solve(x0, u, timer)
        x = self._solver.get("x")
        return x


    def _solve(
        self,
        x0: np.ndarray,
        u: np.ndarray,
        timer: bool,
    ) -> None:
        st = time.perf_counter()
        assert len(x0) == self._nx
        assert len(u) == self._nu

        self._solver.set("x", x0)
        self._solver.set("u", u)
        status = self._solver.solve()
        if status != 0:
            raise Exception(f"acados returned status {status}.")
        if timer:
            print(f"sim runtime: {time.perf_counter() - st}")        


    def _get_dims(
        self,
        model: AcadosModel
    ) -> Tuple[int]:
        nx = model.x.shape[0]
        nu = model.u.shape[0]
        return nx, nu


    def _init_solver(
        self,
        model: AcadosModel,
        sim_step: float,
        control_step: float,
    ) -> AcadosSimSolver:
        sim = AcadosSim()
        sim.model = model
        sim.solver_options.T = control_step
        sim.solver_options.integrator_type = "ERK"
        sim.solver_options.num_stages = 4
        sim.solver_options.num_steps = int(round(control_step / sim_step))
        solver = AcadosSimSolver(sim)
        return solver


    def _clear_files(self) -> None:
        """
        Clean up the acados generated files.
        """
        try: os.remove("acados_sim.json")
        except: print("acados_sim.json")

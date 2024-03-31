#!/usr/bin/python3

from acados_template import AcadosSim, AcadosSimSolver
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import atexit
import os
import shutil
from typing import Tuple
from qrac.models import Quadrotor, DisturbedQuadrotor


class MinimalSim():
    def __init__(
        self,
        model: Quadrotor,
        sim_step: float,
        control_step: float,
        data_len: int,
        axes_min=[-2,-2,0],
        axes_max=[2,2,4]
    ) -> None:
        self._assert(model, sim_step, control_step)
        self._nx = model.nx
        self._nu = model.nu
        self._axmin = axes_min
        self._axmax = axes_max
        self._dt = control_step
        self._len = data_len
        self._anim = 10

        self._x = np.zeros((data_len, model.nx))
        self._u = np.zeros((data_len, model.nu))
        self._k = 0

        self._model = DisturbedQuadrotor(model)
        self._solver = self._init_solver(self._model, sim_step, control_step)
        self._fig, self._ax, self._line = self._init_fig()
        #self._line = self._init_line()[0]
        atexit.register(self._clear_files)

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def nu(self) -> int:
        return self._nu

    def update(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d=np.zeros(12),
        timer=False,
    ) -> np.ndarray:
        u_bd = self._bound_u(u)
        self._solve(x=x, u=u_bd, d=d, timer=timer)
        x_sol = self._solver.get("x")[:self._nx]
        x_bd = self._bound_x(x_sol)
        self._update_data(x_bd, u_bd)
        return x_bd

    def get_xdata(self) -> np.ndarray:
        return self._x

    def get_udata(self) -> np.ndarray:
        return self._u

    def get_plot(self) -> None:
        x = self._x[:,0]
        y = self._x[:,1]
        z = self._x[:,2]

        self._ax.scatter(x[0], y[0], z[0], c="b")
        self._ax.scatter(x[-1], y[-1], z[-1], c="g")
        self._line.set_data_3d(x, y, z)
        plt.show()
        self._reset()

    def get_animation(self, filename=""):
        if self._k > self._len:
            k = int(self._len/self._anim)
        else:
            k = int(self._k/self._anim)
        anim = animation.FuncAnimation(
            fig=self._fig, func=self._animate,
            frames=k, interval=1000*self._dt
        )
        if len(filename):
            print("Saving animation...")
            anim.save(filename=filename, writer="pillow")
            print("Animation saved.")
        plt.show()
        self._reset()

    def _animate(self, i: int):
        x = self._x[:,0]
        y = self._x[:,1]
        z = self._x[:,2]
        self._line.set_data_3d(
            x[:self._anim*i],
            y[:self._anim*i],
            z[:self._anim*i]
        )
        return (self._line)

    def _reset(self) -> None:
        self._ax.clear()
        self._line = self._ax.plot([],[],[], c="r")[0]

    def _init_fig(self) -> Tuple[plt.figure, plt.axes]:
        fig = plt.figure(figsize=(9,8))
        ax = fig.add_subplot(projection="3d")

        ax.set_xlim(self._axmin[0], self._axmax[0])
        ax.set_ylim(self._axmin[1], self._axmax[1])
        ax.set_zlim(self._axmin[2], self._axmax[2])

        ax.xaxis.set_rotate_label(False)
        ax.set_xlabel(r"$\bf{x}$ (m)", fontsize=12)
        ax.yaxis.set_rotate_label(False)
        ax.set_ylabel(r"$\bf{y}$ (m)", fontsize=12)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r"$\bf{z}$ (m)", fontsize=12)

        line = ax.plot([],[],[], c="r")[0]
        return fig, ax, line

    def _update_data(
        self,
        x: np.ndarray,
        u: np.ndarray
    ) -> None:
        self._x[self._k] = x
        self._u[self._k,:] = u
        self._k += 1

    def _bound_u(
        self,
        u: np.ndarray
    ) -> np.ndarray:
        u = np.where(u>self._model.u_min, u, self._model.u_min)
        u = np.where(u<self._model.u_max, u, self._model.u_max)
        return u

    def _bound_x(
        self,
        x: np.ndarray
    ) -> np.ndarray:
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
        assert len(d) == self._nx
        self._solver.set("x", np.concatenate((x,d)))
        self._solver.set("u", u)
        status = self._solver.solve()
        if status != 0:
            raise Exception(f"acados returned status {status}.")
        if timer:
            print(f"sim runtime: {time.perf_counter() - st}")

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
        try:
            shutil.rmtree("c_generated_code")
        except:
            print("failed to delete c_generated_code")
        try:
            os.remove("acados_sim.json")
        except:
            print("acados_sim.json")

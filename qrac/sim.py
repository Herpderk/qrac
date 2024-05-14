#!/usr/bin/python3

import time
import atexit
import os
import shutil
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from acados_template import AcadosSim, AcadosSimSolver, ocp_get_default_cmake_builder
from qrac.models import Quadrotor, DisturbedQuadrotor, PendulumQuadrotor, PendDisturbedQuadrotor


class MinimalSim():
    def __init__(
        self,
        model: Quadrotor,
        sim_step: float,
        control_step: float,
        data_len: int,
        anim_spd=40,
        axes_min=[-1,-1,0],
        axes_max=[1,1,2]
    ) -> None:
        self._assert(model, sim_step, control_step)
        self._nx = model.nx
        self._nu = model.nu
        self._axmin = axes_min
        self._axmax = axes_max
        self._dt = control_step
        self._len = data_len
        self._anim = anim_spd

        self._x = np.zeros((data_len, model.nx))
        self._u = np.zeros((data_len, model.nu))
        self._k = 0

        if type(model) == Quadrotor:
            self._model = DisturbedQuadrotor(model)
        elif type(model) == PendulumQuadrotor:
            #self._model = PendDisturbedQuadrotor(model)
            self._model = model
        self._solver = self._init_solver(self._model, sim_step, control_step)
        self._fig, self._ax, self._traj, self._prop, self._arm0, self._arm1 = self._init_fig()
        
        sprite_size = 0.5
        self._p1 = np.array([sprite_size / 2, 0, 0, 1])
        self._p2 = np.array([-sprite_size / 2, 0, 0, 1])
        self._p3 = np.array([0, sprite_size / 2, 0, 1])
        self._p4 = np.array([0, -sprite_size / 2, 0, 1])
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
        d=None,
        timer=False,
    ) -> np.ndarray:
        if type(d) == type(None):
            d = np.zeros(self._nx)
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

    def get_plot(self, filename="") -> None:
        x = self._x[:,0]
        y = self._x[:,1]
        z = self._x[:,2]

        self._ax.scatter(x[0], y[0], z[0], c="b")
        self._ax.scatter(x[-1], y[-1], z[-1], c="g")
        self._traj.set_data_3d(x, y, z)

        if len(filename):
            print("Saving plot...")
            plt.savefig(filename)
            print("Plot saved.")
        plt.show()
        self._reset()

    def get_animation(self, filename=""):
        if self._k > self._len:
            k = int(self._len/self._anim)
        else:
            k = int(self._k/self._anim)
        anim = animation.FuncAnimation(
            fig=self._fig, func=self._animate_traj,
            frames=k, interval=1000*self._dt
        )
        if len(filename):
            print("Saving animation...")
            anim.save(filename=filename, writer="pillow")
            print("Animation saved.")
        plt.show()
        self._reset()

    def get_quad_animation(self, filename=""):
        if self._k > self._len:
            k = int(self._len/self._anim)
        else:
            k = int(self._k/self._anim)
        anim = animation.FuncAnimation(
            fig=self._fig, func=self._animate_quad,
            frames=k, interval=1000*self._dt
        )
        if len(filename):
            print("Saving animation...")
            anim.save(filename=filename, writer="pillow")
            print("Animation saved.")
        plt.show()
        self._reset()

    def _animate_quad(self, i: int):
        x = self._x[self._anim*i,:]
        p1, p2, p3, p4 = self._get_props(x=x)

        self._prop.set_data_3d(
            [p1[0], p2[0], p3[0], p4[0]],
            [p1[1], p2[1], p3[1], p4[1]],
            [p1[2], p2[2], p3[2], p4[2]],
        )
        self._arm0.set_data_3d(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
        )
        self._arm1.set_data_3d(
            [p3[0], p4[0]],
            [p3[1], p4[1]],
            [p3[2], p4[2]],
        )
        return (self._prop, self._arm0, self._arm1)

    def _get_props(self, x: np.ndarray):
        T = self._get_tf(x=x)
        p1_t = T @ self._p1
        p2_t = T @ self._p2
        p3_t = T @ self._p3
        p4_t = T @ self._p4
        return p1_t, p2_t, p3_t, p4_t

    def _get_tf(self, x: np.ndarray):
        q0 = x[3]
        q1 = x[4]
        q2 = x[5]
        q3 = x[6]
        R = np.array([
            [ 1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2) ],
            [ 2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1) ],
            [ 2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2) ],
        ])
        T = np.block([R, np.reshape(x[:3], (3,1))])
        return T

    def _animate_traj(self, i: int):
        x = self._x[:,0]
        y = self._x[:,1]
        z = self._x[:,2]
        self._traj.set_data_3d(
            x[:self._anim*i],
            y[:self._anim*i],
            z[:self._anim*i]
        )
        return (self._traj)

    def _reset(self) -> None:
        self._ax.clear()
        self._traj = self._ax.plot([],[],[], c="r")[0]

    def _init_fig(self) -> Tuple[plt.figure, plt.axes]:
        fig = plt.figure(figsize=(9,8))
        ax = fig.add_subplot(projection="3d")

        ax.set_xlim(self._axmin[0], self._axmax[0])
        ax.set_ylim(self._axmin[1], self._axmax[1])
        ax.set_zlim(self._axmin[2], self._axmax[2])

        freq = 4
        xspace = (self._axmax[0] - self._axmin[0]) / freq
        yspace = (self._axmax[1] - self._axmin[1]) / freq
        zspace = (self._axmax[2] - self._axmin[2]) / freq
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xspace))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(yspace))
        ax.zaxis.set_major_locator(ticker.MultipleLocator(zspace))

        ax.xaxis.set_rotate_label(False)
        ax.yaxis.set_rotate_label(False)
        ax.zaxis.set_rotate_label(False)
        ax.set_ylabel(r"$\bf{y}$ (m)", fontsize=14)
        ax.set_xlabel(r"$\bf{x}$ (m)", fontsize=14)
        ax.set_zlabel(r"$\bf{z}$ (m)", fontsize=14)

        traj = ax.plot([],[],[], c="r")[0]
        prop = ax.plot([],[], "k.", markersize=25)[0]
        arm0 = ax.plot([],[],[], "b-", linewidth=5)[0]
        arm1 = ax.plot([],[],[], "b-", linewidth=5)[0]
        return fig, ax, traj, prop, arm0, arm1

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

        if type(self._model) != PendulumQuadrotor:
            x = np.hstack((x,d))
        self._solver.set("x", x)
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
        sim.code_export_directory = "sim_c_code"
        sim.model = model.get_acados_model()#get_implicit_model()
        #sim.solver_options.collocation_type = "EXPLICIT_RUNGE_KUTTA" #"GAUSS_RADAU_IIA"
        sim.solver_options.integrator_type = "ERK" #"GNSF"
        sim.solver_options.T = control_step
        sim.solver_options.num_stages = 4
        sim.solver_options.num_steps = int(round(control_step / sim_step))
        '''
        sim.solver_options.newton_iter = 10 # for implicit integrator
        sim.solver_options.sens_forw = True
        sim.solver_options.sens_adj = True
        sim.solver_options.sens_hess = False
        sim.solver_options.sens_algebraic = False
        sim.solver_options.output_z = False
        sim.solver_options.sim_method_jac_reuse = False
        '''

        if os.name == "nt":
            builder = ocp_get_default_cmake_builder()
        else:
            builder = None
        solver = AcadosSimSolver(sim, cmake_builder=builder)
        return solver

    def _assert(
        self,
        model: Quadrotor,
        sim_step: float,
        control_step: float,
    ) -> None:
        if type(model) != Quadrotor and type(model) != PendulumQuadrotor:
            raise TypeError(
                "The inputted model must be of type 'Quadrotor'!")
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
            shutil.rmtree("sim_c_code")
        except:
            print("failed to delete sim_c_code")
        try:
            os.remove("acados_sim.json")
        except:
            print("acados_sim.json")

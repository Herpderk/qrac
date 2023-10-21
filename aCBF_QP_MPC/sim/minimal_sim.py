#!/usr/bin/python3

import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import time


class MinimalSim():
    def __init__(self, backend, controller, x0: np.ndarray, data_len=400, sprite_size=1.5,):
        try:
            backend.update
            backend.nx
            backend.nu
            controller.get_input
            controller.dt
        except AttributeError:
            raise NotImplementedError()
        assert len(x0) == backend.nx
        assert type(data_len) == int
        assert (type(sprite_size) == int or type(sprite_size) == float)

        self._backend = backend
        self._controller = controller
        self._p1 = np.array([sprite_size / 2, 0, 0, 1])
        self._p2 = np.array([-sprite_size / 2, 0, 0, 1])
        self._p3 = np.array([0, sprite_size / 2, 0, 1])
        self._p4 = np.array([0, -sprite_size / 2, 0, 1])

        self._x = mp.Array("f", x0)
        self._x_set = mp.Array("f", x0)
        self._run_flag = mp.Value("b", False)

        self._pose_data = np.empty((data_len, 3))
        self._data_ind = 0
        self._fig, self._ax = self._init_fig()

        self._frontend_proc = mp.Process(target=self._run_frontend, args=[])
        self._backend_proc = mp.Process(target=self._run_backend, args=[])


    def start(self,):
        print("Starting simulator...")
        self._run_flag.value = True
        self._frontend_proc.start()
        self._backend_proc.start()


    def stop(self,):
        self._run_flag.value = False
        self._frontend_proc.join()
        self._backend_proc.join()
        print("Simulator successfully exited.")


    def update_setpoint(self, x_set: np.ndarray):
        assert len(x_set) == self._backend.nx
        self._x_set[:] = x_set


    def _init_fig(self):
        plt.ion()
        fig = plt.figure(figsize=(11,11))
        fig.canvas.mpl_connect("key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None])
        ax = fig.add_subplot(projection="3d")
        return fig, ax


    def _update_data(self, x):
        self._pose_data[self._data_ind] = x[:3]
        if self._data_ind >= len(self._pose_data)-1:
            self._data_ind = 0
        else:
            self._data_ind += 1


    def _get_transform(self, x: np.ndarray):
        Rx = np.array([
            [1,            0,             0],
            [0, np.cos(x[3]), -np.sin(x[3])],
            [0, np.sin(x[3]),  np.cos(x[3])],
        ])
        Ry = np.array([
            [ np.cos(x[4]),  0,  np.sin(x[4])],
            [            0,  1,             0],
            [-np.sin(x[4]),  0,  np.cos(x[4])],
        ])
        Rz = np.array([
            [np.cos(x[5]),    -np.sin(x[5]),    0],
            [np.sin(x[5]),     np.cos(x[5]),    0],
            [          0,               0,      1],
        ])
        R = Rz @ Ry @ Rx
        T = np.block([R, np.reshape(x[:3], (3,1))])
        return T


    def _update_visual(self, x: np.ndarray, timer=False):
        if timer: st = time.perf_counter()

        self._update_data(x)
        T = self._get_transform(x)
        p1_t = T @ self._p1
        p2_t = T @ self._p2
        p3_t = T @ self._p3
        p4_t = T @ self._p4

        plt.cla()
        self._ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                     [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                     [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], "k.", markersize=9)
        self._ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
                     [p1_t[2], p2_t[2]], "b-", linewidth=3)
        self._ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
                     [p3_t[2], p4_t[2]], "b-", linewidth=3)
        self._ax.plot(self._pose_data[:,0], self._pose_data[:,1],
                     self._pose_data[:,2], "r.", markersize=1)

        self._ax.set_xlim(-10,10)
        self._ax.set_ylim(-10,10)
        self._ax.set_zlim(0, 10)
        self._ax.set_xticks(range(-10, 10, 2))
        self._ax.set_yticks(range(-10, 10, 2))
        self._ax.set_zticks(range(0, 11, 2))
        self._ax.set_xlabel("$x$", fontsize=15)
        self._ax.set_ylabel("$y$", fontsize=15)
        self._ax.set_zlabel("$z$", fontsize=15)
        plt.pause(0.00001)
        if timer: print(f"render runtime: {time.perf_counter() - st}")


    def _run_frontend(self,):
        while self._run_flag.value:
            self._update_visual(self._x[:])
        plt.ioff()
        plt.close()


    def _run_backend(self,):
        while self._run_flag.value:
            st = time.perf_counter()
            np_x0 = np.array(self._x[:])
            np_x_set = np.array(self._x_set[:])

            u = self._controller.get_input(x0=np_x0, x_set=np_x_set, timer=True)
            x = self._backend.update(x0=np_x0, u=u, timer=True)
            self._x[:] = x

            print(f"u: {u}")
            print(f"x: {x}")
            while time.perf_counter() - st <= self._controller.dt:
                pass
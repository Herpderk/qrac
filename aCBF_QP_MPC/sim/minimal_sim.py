import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import time


class MinimalSim():
    def __init__(self, sim_backend, controller, x0: np.ndarray, data_len=200, sprite_size=1.5,):
        try:
            sim_backend.run_control_loop
            sim_backend.control_step
            sim_backend.nx
            sim_backend.nu
        except AttributeError:
            raise NotImplementedError()
        assert len(x0) == sim_backend.nx

        self.backend = sim_backend
        self.controller = controller
        self.p1 = np.array([sprite_size / 2, 0, 0, 1])
        self.p2 = np.array([-sprite_size / 2, 0, 0, 1])
        self.p3 = np.array([0, sprite_size / 2, 0, 1])
        self.p4 = np.array([0, -sprite_size / 2, 0, 1])

        self.x = mp.Array("f", x0)
        self.x_set = mp.Array("f", x0)
        self.run_flag = mp.Value("b", False)

        self.pose_data = np.empty((data_len, 3))
        self.data_ind = 0
        self.fig, self.ax = self.init_fig()
        
        self.frontend_proc = mp.Process(target=self.run_frontend, args=[])
        self.backend_proc = mp.Process(target=self.run_backend, args=[])


    def init_fig(self):
        plt.ion()
        fig = plt.figure(figsize=(11,11))
        fig.canvas.mpl_connect('key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        ax = fig.add_subplot(projection="3d")
        return fig, ax


    def update_data(self, x):
        self.pose_data[self.data_ind] = x[:3]
        if self.data_ind >= len(self.pose_data)-1:
            self.data_ind = 0
        else:
            self.data_ind += 1


    def get_transform(self, x: np.ndarray):
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


    def plot(self, x: np.ndarray, timer=False):
        if timer: st = time.perf_counter()
        
        self.update_data(x)
        T = self.get_transform(x)
        p1_t = T @ self.p1
        p2_t = T @ self.p2
        p3_t = T @ self.p3
        p4_t = T @ self.p4

        plt.cla()
        self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                     [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                     [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'k.', markersize=9)
        self.ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
                     [p1_t[2], p2_t[2]], 'b-', linewidth=3)
        self.ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
                     [p3_t[2], p4_t[2]], 'b-', linewidth=3)
        self.ax.plot(self.pose_data[:,0], self.pose_data[:,1],
                     self.pose_data[:,2], 'r.', markersize=1)

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        self.ax.set_zlim(0, 10)
        plt.pause(0.00001)
        if timer: print(f"render runtime: {time.perf_counter() - st}")


    def start(self,):
        print("Starting simulator...")
        self.run_flag.value = True
        self.frontend_proc.start()
        self.backend_proc.start()


    def stop(self,):
        self.run_flag.value = False
        self.frontend_proc.join()
        self.backend_proc.join()
        print("Simulator successfully closed.")


    def set_setpoint(self, x_set: np.ndarray):
        assert len(x_set) == self.backend.nx
        self.x_set[:] = x_set


    def run_frontend(self,):
        while self.run_flag.value:
            self.plot(self.x[:])
        plt.ioff()
        plt.close()


    def run_backend(self,):
        while self.run_flag.value:
            st = time.perf_counter()
            np_x0 = np.array(self.x[:])
            np_x_set = np.array(self.x_set[:])

            u = self.controller.get_next_control(x0=np_x0, x_set=np_x_set, timer=True)
            x = self.backend.run_control_loop(x0=np_x0, u=u, timer=True)
            self.x[:] = x

            print(f"u: {u}")
            print(f"x: {x}")
            while time.perf_counter() - st <= self.backend.control_step:
                pass
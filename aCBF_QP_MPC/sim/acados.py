from acados_template import AcadosSim, AcadosSimSolver, AcadosModel
import matplotlib.pyplot as plt
import numpy as np
import time
import atexit
import os


class SimBackend():
    def __init__(self, model, time_step,):
        assert type(model) == AcadosModel
        self.nx, self.nu = self.get_model_dims(model)
        self.solver = self.init_solver(model, time_step)
        atexit.register(self.delete_compiled_files)

    
    def get_model_dims(self, model):
        nx = model.x.shape[0]
        nu = model.u.shape[0]
        return nx, nu


    def init_solver(self, model, T):
        sim = AcadosSim()
        sim.model = model

        sim.solver_options.T = T
        sim.solver_options.integrator_type = "ERK"
        sim.solver_options.num_stages = 4
        sim.solver_options.num_steps = 100

        solver = AcadosSimSolver(sim)
        return solver


    def run_control_loop(self, x0: np.ndarray, u: np.ndarray, timer=True):
        if timer: st = time.time()
        assert len(x0) == self.nx
        assert len(u) == self.nu

        self.solver.set("x", x0)
        self.solver.set("u", u)
        status = self.solver.solve()
        if status != 0:
            raise Exception(f'acados returned status {status}.')
        
        x0 = self.solver.get("x")
        if timer: print(f"sim runtime: {time.time() - st}")
        return x0
        
    
    def delete_compiled_files(self):
        ''' Clean up the acados generated files.
        '''
        try: os.remove('acados_sim.json')
        except: print('acados_sim.json')


class SimFrontend():
    def __init__(self, x0: np.ndarray, size=0.8, show_animation=True):
        self.p1 = np.array([size / 2, 0, 0, 1]).T
        self.p2 = np.array([-size / 2, 0, 0, 1]).T
        self.p3 = np.array([0, size / 2, 0, 1]).T
        self.p4 = np.array([0, -size / 2, 0, 1]).T

        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.show_animation = show_animation

        if self.show_animation:
            plt.ion()
            fig = plt.figure()
            # for stopping simulation with the esc key.
            fig.canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

            self.ax = fig.add_subplot(111, projection='3d')

        self.update_pose(x0)

    def update_pose(self, x:np.ndarray):
        self.x = x[0]
        self.y = x[1]
        self.z = x[2]
        self.roll = x[3]
        self.pitch = x[4]
        self.yaw = x[5]
        self.x_data.append(x[0])
        self.y_data.append(x[1])
        self.z_data.append(x[2])

        if self.show_animation:
            self.plot()

    def transformation_matrix(self):
        x = self.x
        y = self.y
        z = self.z
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        return np.array(
            [[np.cos(yaw) * np.cos(pitch), -np.sin(yaw) * np.cos(roll) + np.cos(yaw) * np.sin(pitch) * np.sin(roll), np.sin(yaw) * np.sin(roll) + np.cos(yaw) * np.sin(pitch) * np.cos(roll), x],
             [np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(roll) + np.sin(yaw) * np.sin(pitch)
              * np.sin(roll), -np.cos(yaw) * np.sin(roll) + np.sin(yaw) * np.sin(pitch) * np.cos(roll), y],
             [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(yaw), z]
             ])

    def plot(self):  # pragma: no cover
        T = self.transformation_matrix()

        p1_t = T @ self.p1
        p2_t = T @ self.p2
        p3_t = T @ self.p3
        p4_t = T @ self.p4

        plt.cla()

        self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                     [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                     [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'k.')

        self.ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
                     [p1_t[2], p2_t[2]], 'r-')
        self.ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
                     [p3_t[2], p4_t[2]], 'r-')

        self.ax.plot(self.x_data, self.y_data, self.z_data, 'r:')

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        self.ax.set_zlim(0, 10)

        plt.pause(0.001)

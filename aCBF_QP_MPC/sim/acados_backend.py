from acados_template import AcadosSim, AcadosSimSolver, AcadosModel
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import time
import atexit
import os


class AcadosBackend():
    def __init__(self, model, sim_step, control_step,):
        assert type(model) == AcadosModel
        assert control_step > sim_step
        
        self.sim_step = sim_step
        self.control_step = control_step
        self.nx, self.nu = self.get_model_dims(model)
        self.solver = self.init_solver(model, sim_step, control_step)
        atexit.register(self.delete_compiled_files)

    
    def get_model_dims(self, model):
        nx = model.x.shape[0]
        nu = model.u.shape[0]
        return nx, nu


    def init_solver(self, model, sim_step, control_step):
        sim = AcadosSim()
        sim.model = model

        sim.solver_options.T = control_step
        sim.solver_options.integrator_type = "ERK"
        sim.solver_options.num_stages = 4
        sim.solver_options.num_steps = int(round(control_step / sim_step))

        solver = AcadosSimSolver(sim)
        return solver


    def run_control_loop(self, x0: np.ndarray, u: np.ndarray, timer=False):
        if timer: st = time.perf_counter()
        assert len(x0) == self.nx
        assert len(u) == self.nu

        self.solver.set("x", x0)
        self.solver.set("u", u)
        status = self.solver.solve()
        if status != 0:
            raise Exception(f'acados returned status {status}.')
        
        x = self.solver.get("x")
        if timer: print(f"sim runtime: {time.perf_counter() - st}")
        return x

    
    def delete_compiled_files(self):
        ''' Clean up the acados generated files.
        '''
        try: os.remove('acados_sim.json')
        except: print('acados_sim.json')

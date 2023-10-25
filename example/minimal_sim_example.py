#!/usr/bin/python3

from aCBF_QP_MPC.dynamics.acados_model import NonlinearCrazyflie
from aCBF_QP_MPC.control.acados_mpc import AcadosMpc
from aCBF_QP_MPC.sim.acados_backend import AcadosBackend
from aCBF_QP_MPC.sim.minimal_sim import MinimalSim
import numpy as np
import time


def main():
    model = NonlinearCrazyflie(Ax=0, Ay=0, Az=0)

    # initialize controller
    Q = np.diag([8,8,8, 1,1,1, 1,1,1, 1,1,1,])
    R = 0.01 * np.diag([1, 1, 1, 1])
    max_thrust = 0.64           # N
    u_max = max_thrust * np.ones(4)
    u_min = np.zeros(4)
    control_T = 0.02
    N = 6
    mpc = AcadosMpc(
        model=model, Q=Q, R=R, u_max=u_max, u_min=u_min, time_step=control_T, num_nodes=N,)

    # initialize simulator backend
    sim_T = control_T / 100
    backend = AcadosBackend(
        model=model, sim_step=sim_T, control_step=control_T)

    # initialize simulator
    lb_pose = [-5, -5, 0]
    ub_pose = [5, 5, 10]
    sim = MinimalSim(
        backend=backend, controller=mpc, lb_pose=lb_pose, ub_pose=ub_pose,)

    # Run the sim for N control loops
    x0 = np.zeros(12)
    x_set = np.array([-1, 2, 5, 0,0,0, 0,0,0, 0,0,0])

    # You can also run it indefinitely if 'max_steps' is not specified
    while sim.is_alive:
        pass
    sim.start(x0=x0)
    sim.update_setpoint(x_set=x_set)
    time.sleep(5)
    sim.stop()

    #N = 50
    #sim.start(x0=x0, max_steps=N)
    #sim.update_setpoint(x_set=x_set)




if __name__=="__main__":
    main()
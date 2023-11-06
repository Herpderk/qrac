#!/usr/bin/python3

from qrac.dynamics import NonlinearCrazyflie
from qrac.control.acados_mpc import AcadosMpc
from qrac.sim.acados_plant import AcadosPlant
from qrac.sim.minimal_sim import MinimalSim
import numpy as np
import time


def main():
    # get dynamics model and convert to acados
    model = NonlinearCrazyflie(Ax=0, Ay=0, Az=0)

    # initialize controller
    Q = np.diag([6,6,6, 1,1,1, 2,2,2, 1,1,1,])
    R = 0.01 * np.diag([1, 1, 1, 1])
    max_thrust = 0.64           # N
    u_max = max_thrust * np.ones(4)
    u_min = np.zeros(4)
    control_T = 0.02
    N = 16
    rt = False
    mpc = AcadosMpc(
        model=model, Q=Q, R=R, u_max=u_max, u_min=u_min, \
        time_step=control_T, num_nodes=N, real_time=rt,)

    # initialize simulator plant
    sim_T = control_T / 100
    plant = AcadosPlant(
        model=model, sim_step=sim_T, control_step=control_T)

    # initialize simulator
    lb_pose = [-10, -10, 0]
    ub_pose = [10, 10, 10]
    sim = MinimalSim(
        plant=plant, controller=mpc, lb_pose=lb_pose, ub_pose=ub_pose,)

    # define the initial state and setpoint
    x0 = np.zeros(12)
    x_set = np.array([-4, -4, 6, 0,0,0, 0,0,0, 0,0,0])

    # Run the sim for N control loops
    N = int(round(15 / control_T))      # 15 seconds worth of control steps
    sim.start(x0=x0, max_steps=N, verbose=True)
    sim.update_setpoint(x_set=x_set)
    
    # You can also end the sim early with 'stop'
    time.sleep(30)
    sim.stop()


if __name__=="__main__":
    main()
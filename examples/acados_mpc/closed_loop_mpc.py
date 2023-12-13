#!/usr/bin/python3

from qrac.dynamics import Crazyflie
from qrac.trajectory import circle
from qrac.control.acados_mpc import AcadosMpc
from qrac.sim.acados_plant import AcadosPlant
from qrac.sim.minimal_sim import MinimalSim
import numpy as np


def main():
    # get dynamics model
    model = Crazyflie(Ax=0, Ay=0, Az=0)

    # initialize controller
    Q = np.diag([8,8,8, 0.4,0.4,0.4, 2,2,2, 0.4,0.4,0.4,])
    R = np.diag([0, 0, 0, 0])
    max_thrust = 0.64           # N
    u_max = max_thrust * np.ones(4)
    u_min = np.zeros(4)
    control_T = 0.02
    num_nodes = 12
    rt = False
    mpc = AcadosMpc(
        model=model, Q=Q, R=R, u_max=u_max, u_min=u_min, \
        time_step=control_T, num_nodes=num_nodes, real_time=rt,)

    # initialize simulator plant
    sim_T = control_T / 100
    plant = AcadosPlant(
        model=model, sim_step=sim_T, control_step=control_T)

    # initialize simulator
    lb_pose = [-10, -10, 0]
    ub_pose = [10, 10, 10]
    sim = MinimalSim(
        plant=plant, controller=mpc, lb_pose=lb_pose, ub_pose=ub_pose,)


    # define a circular trajectory
    trajectory = circle(r=5, alt=2)

    # Run the sim for N control loops
    x0 = np.zeros(12)
    N = int(round(20 / control_T))      # 20 seconds worth of control loops
    sim.start(x0=x0, max_steps=N, verbose=True)

    # track the given trajectory
    x_set = np.zeros(mpc.n_set)
    dt = mpc.dt
    t0 = sim.timestamp
    while sim.is_alive:
        t = sim.timestamp
        tk = sim.timestamp
        for k in range(num_nodes):
            x_set[k*num_nodes : k*num_nodes + model.nx] = \
                np.array(trajectory(tk - t0))
            tk += dt
        sim.update_setpoint(x_set=x_set)
        while sim.timestamp < t+dt and sim.is_alive:
            pass


if __name__=="__main__":
    main()
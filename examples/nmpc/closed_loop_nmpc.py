#!/usr/bin/python3

from qrac.dynamics import Crazyflie
from qrac.trajectory import Circle
from qrac.control.nmpc import NMPC
from qrac.sim.acados_plant import AcadosPlant
from qrac.sim.minimal_sim import MinimalSim
import numpy as np


def main():
    # get dynamics model
    model = Crazyflie(Ax=0, Ay=0, Az=0)

    # initialize controller
    Q = np.diag([40,40,40, 10,10,10, 20,20,20, 10,10,10])
    R = np.diag([0, 0, 0, 0])
    u_max = model.u_max
    u_min = np.zeros(4)
    mpc_T = 0.006
    num_nodes = 100
    rti = True
    mpc = NMPC(
        model=model, Q=Q, R=R, u_max=u_max, u_min=u_min, \
        time_step=mpc_T, num_nodes=num_nodes, rti=rti, )

    # initialize simulator plant
    sim_T = mpc_T / 10
    plant = AcadosPlant(
        model=model, sim_step=sim_T, control_step=mpc_T)

    # initialize simulator and bounds
    lb_pose = [-10, -10, 0]
    ub_pose = [10, 10, 10]
    sim = MinimalSim(
        plant=plant, controller=mpc,
        lb_pose=lb_pose, ub_pose=ub_pose,)

    # define a circular trajectory
    traj = Circle(v=4, r=4, alt=4)

    # Run the sim for N control loops
    x0 = np.array([4,0,0, 0,0,0, 0,0,0, 0,0,0])
    N = int(round(30 / mpc_T))      # 30 seconds worth of control loops
    sim.start(x0=x0, max_steps=N, verbose=True)

    # track the given trajectory
    xset = np.zeros(mpc.n_set)
    nx = model.nx
    dt = mpc.dt
    t0 = sim.timestamp
    while sim.is_alive:
        t = sim.timestamp
        for k in range(num_nodes):
            xset[k*nx : k*nx + nx] = \
                np.array(traj.get_setpoint(t - t0))
            t += dt
        sim.update_setpoint(xset=xset)


if __name__=="__main__":
    main()
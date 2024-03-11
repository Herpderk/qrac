#!/usr/bin/python3

from qrac.models import Crazyflie, Quadrotor
from qrac.trajectory import Circle
from qrac.control.nmpc import NMPC
from qrac.sim import MinimalSim
import numpy as np


def run():
    SIM_TIME = 30
    SIM_T = 0.0005
    CTRL_T = 0.005
    NODES = 75
    Q = np.diag([40,40,40, 10,10,10, 20,20,20, 10,10,10])
    R = np.diag([0, 0, 0, 0])

    # inaccurate model
    model = Crazyflie(Ax=0, Ay=0, Az=0)

    # initialize mpc
    mpc = NMPC(
        model=model, Q=Q, R=R,
        u_min=model.u_min, u_max=model.u_max, 
        time_step=CTRL_T, num_nodes=NODES,
        rti=True, nlp_max_iter=1, qp_max_iter=5
    )

    # initialize sim
    nx = model.nx
    x0 = np.zeros(nx)
    x0[0] = 8
    steps = int(round(SIM_TIME / CTRL_T))
    sim = MinimalSim(
        x0=x0, model=model,
        sim_step=SIM_T, control_step=CTRL_T,
        data_len=steps
    )

    # define a circular trajectory
    traj = Circle(v=4, r=8, alt=8)

    x = x0
    xset = np.zeros(nx*NODES)
    for k in range(steps):
        t = k*CTRL_T
        for n in range(NODES):
            xset[n*nx : n*nx + nx] = traj.get_setpoint(t)
            t += CTRL_T
        u = mpc.get_input(x=x, xset=xset, timer=True)
        x = sim.update(x=x, u=u, timer=True)

        print(f"\nu: {u}")
        print(f"x: {x}")
        print(f"sim time: {(k+1)*CTRL_T}")

    xdata = sim.get_xdata()


if __name__=="__main__":
    run()

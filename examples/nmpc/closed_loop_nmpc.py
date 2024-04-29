#!/usr/bin/python3

from qrac.models import Crazyflie
from qrac.trajectory import Circle
from qrac.control import NMPC
from qrac.sim import MinimalSim
import numpy as np


def run():
    SIM_TIME = 20
    SIM_T = 0.001
    CTRL_T = 0.01
    NODES = 100
    Q = np.diag([10,10,10, 1,1,1,1, 1,1,1, 1,1,1,])
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
    steps = int(round(SIM_TIME / CTRL_T))
    sim = MinimalSim(
        model=model,
        sim_step=SIM_T, control_step=CTRL_T,
        data_len=steps
    )

    # define a circular trajectory
    traj = Circle(v=2, r=1, alt=1)

    
    x = np.zeros(nx)
    x[3] = 1
    xset = np.zeros(nx*NODES)
    for k in range(steps):
        t = k*CTRL_T
        for n in range(NODES+1):
            xset[n*nx : n*nx + nx] = traj.get_setpoint(t)
            t += CTRL_T
        u = mpc.get_input(x=x, xset=xset, timer=True)
        x = sim.update(x=x, u=u, timer=True)

        print(f"\nu: {u}")
        print(f"x: {x}")
        print(f"sim time: {(k+1)*CTRL_T}")

    sim.get_animation()


if __name__=="__main__":
    run()

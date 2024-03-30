#!/usr/bin/python3

from qrac.models import Crazyflie
from qrac.trajectory import LemniScate
from qrac.control import NMPC
import numpy as np


def run():
    # nmpc settings
    PREDICT_TIME = 20
    CTRL_T = 0.01
    NODES = int(round(PREDICT_TIME / CTRL_T))
    Q = np.diag([4,4,4, 1,1,1, 0,0,0, 0,0,0,])
    R = np.diag([0, 0, 0, 0])


    # traj settings
    A = 5
    B = 1
    AXES = [0,1]
    TRANSLATE = [0,0,5]
    xfilename = "/home/derek/dev/my-repos/qrac/examples/refs/lemniscate/xref.npy"
    ufilename = "/home/derek/dev/my-repos/qrac/examples/refs/lemniscate/uref.npy"


    # init model
    model = Crazyflie(Ax=0, Ay=0, Az=0)


    # init nmpc
    nmpc = NMPC(
        model=model, Q=Q, R=R,
        u_min=model.u_min, u_max=model.u_max, 
        time_step=CTRL_T, num_nodes=NODES,
        rti=False, nlp_max_iter=100, qp_max_iter=50
    )


    # define a trajectory
    traj = LemniScate(
        a=A, b=B, axes=AXES, translation=TRANSLATE
    )


    # run for predefined number of steps
    nx = model.nx
    x = np.zeros(nx)
    xset = np.zeros(nx*NODES)

    for k in range(NODES):
        xset[k*nx : k*nx + nx] = traj.get_setpoint(k*CTRL_T)
    xref, uref = nmpc.get_trajectory(x=x, xset=xset, timer=True, visuals=True)


    # output results
    vavg = np.average(
        np.linalg.norm(xref[:, 6:9], axis=1)
    )
    print(f"Avg velocity: {vavg}")
    np.save(xfilename, xref)
    np.save(ufilename, uref)


if __name__=="__main__":
    run()

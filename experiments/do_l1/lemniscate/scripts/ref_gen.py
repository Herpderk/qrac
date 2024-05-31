#!/usr/bin/python3

import numpy as np
from qrac.models import Crazyflie
from qrac.trajectory import LemniScate
from qrac.control import NMPC


def main():
    # nmpc settings
    PREDICT_TIME = 11
    CTRL_T = 0.01
    NODES = int(round(PREDICT_TIME / CTRL_T))
    Q = np.diag([2,2,2, 1,1,1,1, 1,1,1, 1,1,1,])
    R = np.diag([0, 0, 0, 0])
    D_MAX = np.array([
        0,0,0, 0,0,0,0, 0,0,0, 1,1,1,
    ])

    # traj settings
    A = 1
    B = 1.5
    AXES = [0,1]
    TRANSLATE = [0,0,1]
    xfilename = "../refs/xref.npy"
    ufilename = "../refs/uref.npy"
    dfilename = "../refs/disturb.npy"


    # init model
    model = Crazyflie(Ax=0, Ay=0, Az=0)


    # init nmpc
    nmpc = NMPC(
        model=model, Q=Q, R=R,
        u_min=model.u_min, u_max=model.u_max,
        time_step=CTRL_T, num_nodes=NODES,
        rti=False, nlp_max_iter=1000, qp_max_iter=1000
    )


    # define a trajectory
    traj = LemniScate(
        a=A, b=B, axes=AXES, translation=TRANSLATE
    )


    # run for predefined number of steps
    nx = model.nx
    x = np.zeros(nx)
    x[3] = 1
    xset = np.zeros(nx*NODES)

    for k in range(NODES):
        xset[k*nx : k*nx + nx] = traj.get_setpoint(k*CTRL_T)
    xref, uref = nmpc.get_trajectory(x=x, xset=xset, timer=True, visuals=True)


    # output results
    vavg = np.average(
        np.linalg.norm(xref[:, 7:10], axis=1)
    )
    print(f"Avg velocity: {vavg}")
    np.save(xfilename, xref)
    np.save(ufilename, uref)
    
    d = D_MAX * np.random.uniform(-1, 1, (NODES, nx))
    np.save(dfilename, d)


if __name__=="__main__":
    main()

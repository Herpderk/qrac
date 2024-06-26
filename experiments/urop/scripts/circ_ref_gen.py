#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from qrac.models import Crazyflie
from qrac.trajectory import Circle
from qrac.control import NMPC
from consts import PREDICT_TIME, CTRL_T, Q_TRAJ, R_TRAJ


def main():
    NODES = int(round(PREDICT_TIME / CTRL_T))


    # traj settings
    V = 1
    RAD = 1
    ALT = 1
    xfilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/circ_xref.npy"
    ufilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/circ_uref.npy"
    figname = "/home/derek/dev/my-repos/qrac/experiments/urop/figures/circ_xref.png"


    # init model
    model = Crazyflie(Ax=0, Ay=0, Az=0)


    # init nmpc
    nmpc = NMPC(
        model=model, Q=Q_TRAJ, R=R_TRAJ,
        u_min=model.u_min, u_max=model.u_max,
        time_step=CTRL_T, num_nodes=NODES,
        rti=False, nlp_max_iter=200, qp_max_iter=100
    )


    # define a trajectory
    traj = Circle(
        v=V, r=RAD, alt=ALT
    )


    # run for predefined number of steps
    nx = model.nx
    x = np.zeros(nx)
    x[3] = 1
    xset = np.zeros(nx*NODES)

    for k in range(NODES):
        xset[k*nx : k*nx + nx] = traj.get_setpoint(k*CTRL_T)
    xref, uref = nmpc.get_trajectory(x=x, xset=xset, timer=True, visuals=True)


    # average velocity
    vavg = np.average(
        np.linalg.norm(xref[:, nx-6 : nx-3], axis=1)
    )
    print(f"Avg velocity: {vavg}")


    np.save(xfilename, xref)
    np.save(ufilename, uref)


    # plot xref
    fig = plt.figure(figsize=(9,8))
    ax = fig.add_subplot(projection="3d")

    ax.set_xlim(-RAD, RAD)
    ax.set_ylim(-RAD, RAD)
    ax.set_zlim(0, 2*RAD)
    ax.xaxis.set_rotate_label(False)

    freq = 4
    xspace = (2*RAD) / freq
    yspace = (2*RAD) / freq
    zspace = (2*ALT) / freq
    ax.xaxis.set_major_locator(ticker.MultipleLocator(xspace))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(yspace))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(zspace))

    ax.set_xlabel(r"$\bf{x}$ (m)", fontsize=12)
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel(r"$\bf{y}$ (m)", fontsize=12)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r"$\bf{z}$ (m)", fontsize=12)

    xref = np.load(xfilename)
    x = xref[:,0]
    y = xref[:,1]
    z = xref[:,2]
    line = ax.plot([],[],[], c="r")[0]
    line.set_data_3d(x, y, z)

    plt.savefig(figname)
    plt.show()


if __name__=="__main__":
    main()

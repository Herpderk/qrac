#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from qrac.models import Crazyflie
from qrac.trajectory import Circle
from qrac.control import NMPC


def main():
    # nmpc settings
    PREDICT_TIME = 30
    CTRL_T = 0.01
    NODES = int(round(PREDICT_TIME / CTRL_T))
    Q = np.diag([20,20,20, 1,1,1, 1,1,1, 1,1,1,])
    R = np.diag([0, 0, 0, 0])


    # traj settings
    ALT = 1
    xfilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_xref.npy"
    ufilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_uref.npy"
    figname = "/home/derek/dev/my-repos/qrac/experiments/urop/figures/setpt_xref.png"


    # init model
    model = Crazyflie(Ax=0, Ay=0, Az=0)


    # init nmpc
    nmpc = NMPC(
        model=model, Q=Q, R=R,
        u_min=model.u_min, u_max=model.u_max,
        time_step=CTRL_T, num_nodes=NODES,
        rti=False, nlp_max_iter=200, qp_max_iter=100
    )


    # define a setpoint
    setpt = np.zeros(12)
    setpt[2] = ALT


    # run for predefined number of steps
    nx = model.nx
    x = setpt
    xset = np.zeros(nx*NODES)

    for k in range(NODES):
        xset[k*nx : k*nx + nx] = setpt
    xref, uref = nmpc.get_trajectory(x=x, xset=xset, timer=True, visuals=True)


    np.save(xfilename, xref)
    np.save(ufilename, uref)


    # plot xref
    fig = plt.figure(figsize=(9,8))
    ax = fig.add_subplot(projection="3d")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 2)
    ax.xaxis.set_rotate_label(False)

    freq = 4
    xspace = (2) / freq
    yspace = (2) / freq
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

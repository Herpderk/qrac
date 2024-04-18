#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from qrac.models import Crazyflie, Quadrotor
from qrac.control import NMPC
from consts import PREDICT_TIME, CTRL_T, Q_TRAJ, R_TRAJ


def main():
    NODES = int(round(PREDICT_TIME / CTRL_T))


    # traj settings
    ALT = 1
    xfilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_xref.npy"
    ufilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_uref.npy"
    figname = "/home/derek/dev/my-repos/qrac/experiments/urop/figures/setpt_xref.png"


    # accurate model
    acc = Crazyflie(Ax=0, Ay=0, Az=0)

    # inaccurate model
    m_inacc = 2 * acc.m
    Ixx_inacc = 10 * acc.Ixx
    Iyy_inacc = 10 * acc.Iyy
    Izz_inacc = 10 * acc.Izz
    Ax_inacc = 0
    Ay_inacc = 0
    Az_inacc = 0
    xB_inacc = acc.xB
    yB_inacc = acc.yB
    kf_inacc = acc.kf
    km_inacc = acc.km
    u_min_inacc =acc.u_min
    u_max_inacc =acc.u_max
    inacc = Quadrotor(
        m=m_inacc, Ixx=Ixx_inacc,Iyy=Iyy_inacc, Izz=Izz_inacc,
        Ax=Ax_inacc, Ay=Ay_inacc, Az=Az_inacc, kf=kf_inacc, km=km_inacc,
        xB=xB_inacc, yB=yB_inacc, u_min=u_min_inacc, u_max=u_max_inacc
    )


    # init nmpc
    nmpc = NMPC(
        model=inacc, Q=Q_TRAJ, R=R_TRAJ,
        u_min=inacc.u_min, u_max=inacc.u_max,
        time_step=CTRL_T, num_nodes=NODES,
        rti=False, nlp_max_iter=200, qp_max_iter=100
    )


    # define a setpoint
    nx = inacc.nx
    setpt = np.zeros(nx)
    setpt[2] = ALT
    setpt[3] = 1


    # run for predefined number of steps
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

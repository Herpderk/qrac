#!/usr/bin/python3

import numpy as np
from qrac.models import Quadrotor, Crazyflie
from qrac.control import NMPC
from qrac.control import L1Augmentation
from qrac.sim import MinimalSim


def main():
    # mpc settings
    CTRL_T = 0.01
    NODES = 40
    MAX_ITER_NMPC = 5
    Q = np.diag([4,4,4, 2,2,2, 1,1,1, 1,1,1,])
    R = np.diag([0, 0, 0, 0])

    # L1 settings
    A_GAIN = 20
    W = 100

    # sim settings
    SIM_T = CTRL_T / 10

    # file access
    xfilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_xref.npy"
    ufilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_uref.npy"
    dfilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_d.npy"
    trajname = "/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_l1_traj.npy"
    figname = "/home/derek/dev/my-repos/qrac/experiments/urop/figures/setpt_l1.png"


    # load in references and disturbance
    xref = np.load(xfilename)
    uref = np.load(ufilename)
    disturb = np.load(dfilename)

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


    # init mpc
    mpc = NMPC(
        model=inacc, Q=Q, R=R,
        u_min=inacc.u_min, u_max=inacc.u_max,
        time_step=CTRL_T, num_nodes=NODES,
        rti=True, nlp_max_iter=1, qp_max_iter=MAX_ITER_NMPC
    )

    # init L1
    l1_mpc = L1Augmentation(
        model=inacc, control_ref=mpc,
        adapt_gain=A_GAIN, bandwidth=W,
    )


    # init sim
    steps = xref.shape[0]
    sim = MinimalSim(
        model=acc, data_len=steps,
        sim_step=SIM_T, control_step=CTRL_T,
    )


    # run for predefined number of steps
    nx = inacc.nx
    xset = np.zeros(nx*NODES)
    x = xref[0]
    nu = inacc.nu
    uset = np.zeros(nu*NODES)

    for k in range(steps):
        diff = steps - k
        if diff < NODES:
            xset[:nx*diff] = xref[k : k+diff, :].flatten()
            xset[:nx*(NODES-diff)] = np.tile(xref[-1,:], NODES-diff)
            uset[:nu*diff] = uref[k : k+diff, :].flatten()
            uset[:nu*(NODES-diff)] = np.tile(uref[-1,:], NODES-diff)

        else:
            xset[:] = xref[k : k+NODES, :].flatten()
            uset[:nu*NODES] = uref[k : k+NODES, :].flatten()

        u = l1_mpc.get_input(x=x, xset=xset, uset=uset, timer=True)
        d = disturb[k,:]
        x = sim.update(x=x, u=u, d=d, timer=True)

        print(f"\nu: {u}")
        print(f"x: {x}")
        print(f"sim time: {(k+1)*CTRL_T}\n")


    # save trajectory data
    xdata = sim.get_xdata()
    np.save(trajname, xdata)


    # calculate RMSE
    err = np.sum(np.abs( xref[:,0:3] - xdata[:,0:3] )**2,axis=-1)**0.5
    rmse = np.sqrt(np.sum(np.square(err))/err.shape[0])
    print(f"RMSE: {rmse}")


    # plot
    sim.get_plot(figname)


if __name__=="__main__":
    main()

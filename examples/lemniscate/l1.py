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
    Q = np.diag([1,1,1, 1,1,1,1, 1,1,1, 1,1,1,])
    R = np.diag([0, 0, 0, 0])

    # L1 settings
    A_GAIN = 10000
    W = 1000

    # sim settings
    SIM_T = CTRL_T / 10

    # file access
    xfilename = "../refs/lemniscate/xref.npy"
    ufilename = "../refs/lemniscate/uref.npy"
    dfilename = "../refs/lemniscate/disturb.npy"

    # load in time optimal trajectory
    xref = np.load(xfilename)
    uref = np.load(ufilename)
    disturb = np.load(dfilename)

    # inaccurate model
    inacc = Crazyflie(Ax=0, Ay=0, Az=0)

    # true model
    m_true = 1 * inacc.m
    Ixx_true = 1 * inacc.Ixx
    Iyy_true = 1 * inacc.Iyy
    Izz_true = 1 * inacc.Izz
    Ax_true = 0
    Ay_true = 0
    Az_true = 0
    xB_true = inacc.xB
    yB_true = inacc.yB
    kf_true = inacc.kf
    km_true = inacc.km
    u_min_true =inacc.u_min
    u_max_true =inacc.u_max
    acc = Quadrotor(
        m=m_true, Ixx=Ixx_true,Iyy=Iyy_true, Izz=Izz_true,
        Ax=Ax_true, Ay=Ay_true, Az=Az_true, kf=kf_true, km=km_true,
        xB=xB_true, yB=yB_true, u_min=u_min_true, u_max=u_max_true
    )

    # init mpc
    mpc = NMPC(
        model=inacc, Q=Q, R=R,
        u_min=inacc.u_min, u_max=inacc.u_max,
        time_step=CTRL_T, num_nodes=NODES,
        rti=True, nlp_max_iter=1, qp_max_iter=10
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
    sat = 0.0
    utot = np.zeros(nu)

    for k in range(steps):
        diff = steps - k
        if diff < NODES:
            xset[:nx*diff] = xref[k : k+diff, :].flatten()
            xset[:nx*(NODES-diff)] = np.tile(xref[-1,:], NODES-diff)
            uset[:nu*diff] = uref[k : k+diff, :].flatten()
            uset[:nu*(NODES-diff)] = np.tile(uref[-1,:], NODES-diff)

        else:
            xset[:] = xref[k:k+NODES, :].flatten()
            uset[:nu*NODES] = uref[k:k+NODES, :].flatten()

        u = l1_mpc.get_input(x=x, xset=xset, uset=uset, timer=True)
        utot += u
        for inp in u:
            if inp > 0.15:
                sat += inp - 0.15
                break
        d = 10*np.hstack(
            (np.zeros(9), disturb[k,-4:])
        )
        x = sim.update(x=x, u=u, d=d, timer=True)

        print(f"\nu: {u}")
        print(f"x: {x}")
        print(f"sim time: {(k+1)*CTRL_T}\n")

    # calculate RMSE
    xdata = sim.get_xdata()
    err = np.sum(( xref[:,0:3] - xdata[:,0:3] )**2,axis=-1)
    rmse = np.sqrt(np.sum(err)/err.shape[0])
    print(f"root mean square error: {rmse}")
    print(f"total saturation: {sat}")
    print(f"average saturation: {sat/steps}")
    print(f"average saturation per actuator: {sat/4/steps}")
    print(f"average u: {np.sum(utot) / 4 / steps}")

    # plot
    sim.get_animation()


if __name__=="__main__":
    main()

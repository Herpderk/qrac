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
    Q = np.diag([4,4,4, 2,2,2,2, 1,1,1, 1,1,1,])
    R = np.diag([0, 0, 0, 0])

    # L1 settings
    A_GAIN = 200
    W = 80

    # L1 optimizer settings
    M = 10
    A_GAIN_MIN = 0
    A_GAIN_MAX = 800
    W_MIN = 0
    W_MAX = 800

    # sim settings
    SIM_T = CTRL_T / 10
    D_MAX = np.array([
        0,0,0, 0,0,0,0, 5,5,5, 5,5,5,
    ])

    # file access
    xfilename = "../refs/circle/xref.npy"
    ufilename = "../refs/circle/uref.npy"

    # load in time optimal trajectory
    xref = np.load(xfilename)
    uref = np.load(ufilename)


    # inaccurate model
    inacc = Crazyflie(Ax=0, Ay=0, Az=0)

    # true model
    m_true = 1.5 * inacc.m
    Ixx_true = 5 * inacc.Ixx
    Iyy_true = 5 * inacc.Iyy
    Izz_true = 5 * inacc.Izz
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
        opt_horizon=M,
        adapt_gain_min=A_GAIN_MIN,
        adapt_gain_max=A_GAIN_MAX,
        bandwidth_min=W_MIN, bandwidth_max=W_MAX,
        u_min=inacc.u_min, u_max=inacc.u_max,
        rti=True, nlp_tol=10**(-6), nlp_max_iter=1, qp_max_iter=5
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
            xset[:] = xref[k:k+NODES, :].flatten()
            uset[:nu*NODES] = uref[k:k+NODES, :].flatten()

        u = l1_mpc.get_input(x=x, xset=xset, uset=uset, timer=True)
        #d = D_MAX*np.random.uniform(-1, 1, nx)
        d = np.zeros(13)
        x = sim.update(x=x, u=u, d=d, timer=True)

        print(f"\nu: {u}")
        print(f"x: {x}")
        print(f"sim time: {(k+1)*CTRL_T}\n")


    # calculate RMSE
    res = 0
    xdata = sim.get_xdata()
    for k in range(steps-1):
        res += np.linalg.norm(
            xref[k, 0:3] - xdata[k+1, 0:3], ord=2
        )
    rmse = np.sqrt(res/steps)
    print(f"root mean square error: {rmse}")


    # plot
    sim.get_animation()


if __name__=="__main__":
    main()

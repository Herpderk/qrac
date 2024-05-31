#!/usr/bin/python3

import numpy as np
from qrac.models import Quadrotor, Crazyflie, PendulumQuadrotor
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
    A_GAIN = 50000
    W = 1000

    # L1 optimizer settings
    M = 1
    W_MIN = 10**-6
    W_MAX = 1000

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

    # accurate model
    m_pend = 1*inacc.m
    l_pend = 40*np.abs(inacc.xB[0])
    acc = PendulumQuadrotor(
        model=inacc, m_pend=m_pend, l_pend=l_pend
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
        adapt_gain=A_GAIN, bandwidth=W, opt_horizon=M,
        bandwidth_min=W_MIN, bandwidth_max=W_MAX,
        rti=True, nlp_tol=10**(-2), nlp_max_iter=1, qp_max_iter=400
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
    x = np.zeros(acc.nx)
    x[:nx] = xref[0]
    nu = inacc.nu
    uset = np.zeros(nu*NODES)
    wb = np.zeros(steps)

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

        u = l1_mpc.get_input(x=x[:nx], xset=xset, uset=uset, timer=True)
        wb[k] = l1_mpc.get_bandwidth()
        d = disturb[k,:]
        d = np.zeros(acc.nx)
        x = sim.update(x=x, u=u, d=d, timer=True)

        print(f"\nu: {u}")
        print(f"x: {x}")
        print(f"sim time: {(k+1)*CTRL_T}\n")


    # calculate RMSE
    xdata = sim.get_xdata()
    err = np.sum(( xref[:,0:3] - xdata[:,0:3] )**2,axis=-1)**0.5
    rmse = np.sqrt(np.sum(np.square(err))/err.shape[0])
    print(f"root mean square error: {rmse}")

    # plot
    sim.get_animation()
    
    # save bandwidth history
    np.save(f"bandwidths_N={M}", wb)
    np.save(f"traj_N={M}", xdata)


if __name__=="__main__":
    main()

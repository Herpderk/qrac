#!/usr/bin/python3

import numpy as np
from qrac.models import Quadrotor, Crazyflie, PendulumQuadrotor
from qrac.control import NMPC
from qrac.sim import MinimalSim


def main():
    # mpc settings
    CTRL_T = 0.01
    NODES = 100
    Q = np.diag([4,4,4, 2,2,2,2, 1,1,1, 1,1,1,])
    R = np.diag([0, 0, 0, 0])

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
    m_pend = 0.8*inacc.m
    l_pend = np.abs(inacc.xB[0])
    acc = PendulumQuadrotor(
        model=inacc, m_pend=m_pend, l_pend=l_pend
    )

    # init mpc
    mpc = NMPC(
        model=inacc, Q=Q, R=R,
        u_min=inacc.u_min, u_max=inacc.u_max,
        time_step=CTRL_T, num_nodes=NODES,
        rti=True, nlp_max_iter=1, qp_max_iter=5
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

        u = mpc.get_input(x=x[:nx], xset=xset, uset=uset, timer=True)
        #d = D_MAX*np.random.uniform(-1, 1, nx)
        d = np.zeros(acc.nx)
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

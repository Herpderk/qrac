#!/usr/bin/python3

import numpy as np
from qrac.models import Crazyflie, Quadrotor
from qrac.control import NMPC
from qrac.sim import MinimalSim


def run():
    # mpc settings
    CTRL_T = 0.01
    NODES = 150
    Q = np.diag([1,1,1, 1,1,1, 1,1,1, 1,1,1,])
    R = np.diag([0, 0, 0, 0])

    # sim settings
    SIM_T = CTRL_T / 10
    D_RAND = np.array([
        0,0,0, 0,0,0, 10,10,10, 5,5,5,
    ])

    # reference trajectory files
    xfilename = "/home/derek/dev/my-repos/qrac/examples/refs/circle/xref.npy"
    ufilename = "/home/derek/dev/my-repos/qrac/examples/refs/circle/uref.npy"


    # load in time optimal trajectory
    xref = np.load(xfilename)
    uref = np.load(ufilename)


    # inaccurate model
    inacc = Crazyflie(Ax=0, Ay=0, Az=0)

    # true model
    m_true = 1.5 * inacc.m
    Ixx_true = 20 * inacc.Ixx
    Iyy_true = 20 * inacc.Iyy
    Izz_true = 20 * inacc.Izz
    Ax_true = 0
    Ay_true = 0
    Az_true = 0
    xB_true =inacc.xB
    yB_true =inacc.yB
    k_true =inacc.k
    u_min_true =inacc.u_min
    u_max_true =inacc.u_max
    acc = Quadrotor(
        m_true, Ixx_true, Iyy_true, Izz_true,
        Ax_true, Ay_true, Az_true,
        xB_true, yB_true, k_true,
        u_min_true, u_max_true)


    # init mpc
    nmpc = NMPC(
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

        u = nmpc.get_input(x=x, xset=xset, uset=uset, timer=True)
        d = 2*D_RAND*(-0.5 + np.random.rand(12))
        x = sim.update(x=x, u=u, d=d, timer=True)

        print(f"\nu: {u}")
        print(f"x: {x}")
        print(f"sim time: {(k+1)*CTRL_T}\n")


    # calculate RMSE
    res = 0
    xdata = sim.get_xdata()
    for k in range(steps):
        res += np.linalg.norm(
            xref[k, 0:3] - xdata[k, 0:3], ord=2
        )
    rmse = np.sqrt(res/steps)
    print(f"root mean square error: {rmse}")


    # plot
    sim.get_animation(
        filename=f"/home/derek/Documents/qrac/circle/nmpc_circle_{rmse}.gif"
    )


if __name__=="__main__":
    run()

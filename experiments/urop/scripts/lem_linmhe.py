#!/usr/bin/python3

import numpy as np
from qrac.models import Crazyflie, Quadrotor, AffineQuadrotor
from qrac.control import AdaptiveNMPC
from qrac.estimation import MHE
from qrac.sim import MinimalSim


def main():
    # mpc settings
    CTRL_T = 0.01
    NODES = 40
    MAX_ITER_NMPC = 5
    Q = np.diag([4,4,4, 2,2,2, 1,1,1, 1,1,1,])
    R = np.diag([0, 0, 0, 0])

    # estimator settings
    Q_MHE = 1 * np.diag([1, 1,1,1, 1,1,1,])
    R_MHE = 1 * np.diag([1,1,1, 1,1,1, 1,1,1, 1,1,1])
    NODES_MHE = 20
    MAX_ITER_MHE = 5
    D_MAX = np.array([
        0,0,0, 0,0,0, 4,4,4, 4,4,4,
    ])
    D_MIN = -D_MAX

    # sim settings
    SIM_T = CTRL_T / 10

    # file access
    xfilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/lem_xref.npy"
    ufilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/lem_uref.npy"
    dfilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/lem_d.npy"
    trajname = "/home/derek/dev/my-repos/qrac/experiments/urop/data/lem_linmhe_traj.npy"
    figname = "/home/derek/dev/my-repos/qrac/experiments/urop/figures/lem_linmhe.png"


    # load in time optimal trajectory
    xref = np.load(xfilename)
    uref = np.load(ufilename)
    disturb = np.load(dfilename)


    # inaccurate model
    inacc = Crazyflie(Ax=0, Ay=0, Az=0)

    # true model
    m_true = 2 * inacc.m
    Ixx_true = 4 * inacc.Ixx
    Iyy_true = 4 * inacc.Iyy
    Izz_true = 4 * inacc.Izz
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

   
    # define realistic parameter bounds
    # moments of inertia bounds based on max moment of x5
    p_min = np.zeros(7)
    p_min[0] = 1 / (2*inacc.m)
    p_min[1] = 1 / (4*inacc.Ixx)
    p_min[2] = 1 / (4*inacc.Iyy)
    p_min[3] = 1 / (4*inacc.Izz)
    p_min[4] = (inacc.Izz - 4*inacc.Iyy) / inacc.Ixx
    p_min[5] = (inacc.Ixx - 4*inacc.Izz) / inacc.Iyy
    p_min[6] = (inacc.Iyy - 4*inacc.Ixx) / inacc.Izz

    p_max = np.zeros(7)
    p_max[0] = 1 / inacc.m
    p_max[1] = 1 / (inacc.Ixx)
    p_max[2] = 1 / (inacc.Iyy)
    p_max[3] = 1 / (inacc.Iyy)
    p_max[4] = (4*inacc.Izz - inacc.Iyy) / inacc.Ixx
    p_max[5] = (4*inacc.Ixx - inacc.Izz) / inacc.Iyy
    p_max[6] = (4*inacc.Iyy - inacc.Ixx) / inacc.Izz


    # init estimator
    mhe = MHE(
        model=inacc, Q=Q_MHE, R=R_MHE,
        param_min=p_min, param_max=p_max,
        disturb_min=D_MIN, disturb_max=D_MAX,
        time_step=CTRL_T, num_nodes=NODES_MHE,
        rti=True, nonlinear=False,
        nlp_max_iter=1, qp_max_iter=MAX_ITER_MHE
    )


    # init mpc
    anmpc = AdaptiveNMPC(
        model=inacc, estimator=mhe, Q=Q, R=R,
        u_min=inacc.u_min, u_max=inacc.u_max, time_step=CTRL_T,
        num_nodes=NODES, real_time=False, rti=True,
        nlp_max_iter=1, qp_max_iter=MAX_ITER_NMPC
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

        u = anmpc.get_input(x=x, xset=xset, uset=uset, timer=True)
        d = disturb[k,:]
        x = sim.update(x=x, u=u, d=d, timer=True)

        print(f"\nu: {u}")
        print(f"x: {x}")
        print(f"sim time: {(k+1)*CTRL_T}\n")

    print(f"init param min: \n{p_min}")
    print(f"init param max: \n{p_max}")
    print(f"acc params:\n{AffineQuadrotor(acc).get_parameters()}")
    print(f"inacc params:\n{AffineQuadrotor(inacc).get_parameters()}")


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

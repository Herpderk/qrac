#!/usr/bin/python3

import numpy as np
from qrac.models import Crazyflie, Quadrotor, AffineQuadrotor
from qrac.control import AdaptiveNMPC
from qrac.estimation import MHE
from qrac.sim import MinimalSim
from consts import CTRL_T, SIM_T, NODES, MAX_ITER_NMPC, Q, R,\
                    Q_MHE_LIN, R_MHE, D_MAX, D_MIN, NODES_MHE, MAX_ITER_MHE


def main():
    # file access
    xfilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_xref.npy"
    ufilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_uref.npy"
    dfilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_d.npy"
    trajname = "/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_linmhe_traj.npy"
    figname = "/home/derek/dev/my-repos/qrac/experiments/urop/figures/setpt_linmhe.png"


    # load in time optimal trajectory
    xref = np.load(xfilename)
    uref = np.load(ufilename)
    disturb = np.load(dfilename)


    # accurate model
    acc = Crazyflie(Ax=0.01, Ay=0.01, Az=0.02)

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

    # realistic parameter bounds
    p_min = np.zeros(10)
    p_min[0] = 1 / (2*acc.m)
    p_min[1:4] = np.zeros(3)
    p_min[4] = 1 / (20*acc.Ixx)
    p_min[5] = 1 / (20*acc.Iyy)
    p_min[6] = 1 / (20*acc.Izz)
    p_min[7] = (acc.Izz - 20*acc.Iyy) / acc.Ixx
    p_min[8] = (acc.Ixx - 20*acc.Izz) / acc.Iyy
    p_min[9] = (acc.Iyy - 20*acc.Ixx) / acc.Izz

    p_max = np.zeros(10)
    p_max[0] = 1 / acc.m
    p_max[1:4] = 0.5 * np.ones(3)
    p_max[4] = 1 / (acc.Ixx)
    p_max[5] = 1 / (acc.Iyy)
    p_max[6] = 1 / (acc.Iyy)
    p_max[7] = (20*acc.Izz - acc.Iyy) / acc.Ixx
    p_max[8] = (20*acc.Ixx - acc.Izz) / acc.Iyy
    p_max[9] = (20*acc.Iyy - acc.Ixx) / acc.Izz


    # init estimator
    mhe = MHE(
        model=inacc, Q=Q_MHE_LIN, R=R_MHE,
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

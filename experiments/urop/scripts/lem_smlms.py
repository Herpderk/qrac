#!/usr/bin/python3

import numpy as np
from qrac.models import Crazyflie, Quadrotor, AffineQuadrotor
from qrac.control import AdaptiveNMPC
from qrac.estimation import SetMembershipEstimator, LMS
from qrac.sim import MinimalSim
from consts import CTRL_T, SIM_T, Q, R, NODES, MAX_ITER_NMPC,\
                    D_MAX, D_MIN, U_GAIN, P_TOL, MAX_ITER_SM


def main():
    # file access
    xfilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/lem_xref.npy"
    ufilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/lem_uref.npy"
    dfilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/lem_d.npy"
    trajname = "/home/derek/dev/my-repos/qrac/experiments/urop/data/lem_smlms_traj.npy"
    figname = "/home/derek/dev/my-repos/qrac/experiments/urop/figures/lem_smlms.png"


    # load in references and disturbance
    xref = np.load(xfilename)
    uref = np.load(ufilename)
    disturb = np.load(dfilename)


    # inaccurate model
    inacc = Crazyflie(Ax=0, Ay=0, Az=0)

    # true model
    m_true = 2 * inacc.m
    Ixx_true = 10 * inacc.Ixx
    Iyy_true = 10 * inacc.Iyy
    Izz_true = 10 * inacc.Izz
    Ax_true = 0.01
    Ay_true = 0.01
    Az_true = 0.02
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
    p_min = np.zeros(10)
    p_min[0] = 1 / (2*inacc.m)
    p_min[1:4] = np.zeros(3)
    p_min[4] = 1 / (20*inacc.Ixx)
    p_min[5] = 1 / (20*inacc.Iyy)
    p_min[6] = 1 / (20*inacc.Izz)
    p_min[7] = (inacc.Izz - 20*inacc.Iyy) / inacc.Ixx
    p_min[8] = (inacc.Ixx - 20*inacc.Izz) / inacc.Iyy
    p_min[9] = (inacc.Iyy - 20*inacc.Ixx) / inacc.Izz

    p_max = np.zeros(10)
    p_max[0] = 1 / inacc.m
    p_max[1:4] = 0.5 * np.ones(3)
    p_max[4] = 1 / (inacc.Ixx)
    p_max[5] = 1 / (inacc.Iyy)
    p_max[6] = 1 / (inacc.Iyy)
    p_max[7] = (20*inacc.Izz - inacc.Iyy) / inacc.Ixx
    p_max[8] = (20*inacc.Ixx - inacc.Izz) / inacc.Iyy
    p_max[9] = (20*inacc.Iyy - inacc.Ixx) / inacc.Izz


    # init LMS
    lms = LMS(
        model=inacc, param_min=p_min, param_max=p_max,
        update_gain=U_GAIN, time_step=CTRL_T
    )

    # init set-membership
    sm = SetMembershipEstimator(
        model=inacc, estimator=lms,
        param_tol=P_TOL, param_min=p_min, param_max=p_max,
        disturb_min=D_MIN, disturb_max=D_MAX, time_step=CTRL_T,
        qp_tol=10**-6, max_iter=MAX_ITER_SM
    )

    # init adaptive mpc
    anmpc = AdaptiveNMPC(
        model=inacc, estimator=sm, Q=Q, R=R,
        u_min=inacc.u_min, u_max=inacc.u_max, time_step=CTRL_T,
        num_nodes=NODES, rti=True, real_time=False,
        nlp_tol=10**-6, nlp_max_iter=1, qp_max_iter=MAX_ITER_NMPC
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


    # save plot
    sim.get_plot(figname)
    #sim.get_animation("/home/derek/dev/my-repos/qrac/experiments/urop/figures/lem_anim.gif")


if __name__=="__main__":
    main()

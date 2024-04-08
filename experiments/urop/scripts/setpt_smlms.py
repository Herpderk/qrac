#!/usr/bin/python3

import numpy as np
from qrac.models import Crazyflie, Quadrotor, AffineQuadrotor
from qrac.control import AdaptiveNMPC
from qrac.estimation import SetMembershipEstimator, LMS
from qrac.sim import MinimalSim


def main():
    # mpc settings
    CTRL_T = 0.01
    NODES = 40
    MAX_ITER_NMPC = 5
    Q = np.diag([4,4,4, 2,2,2, 1,1,1, 1,1,1,])
    R = np.diag([0, 0, 0, 0])

    # estimator settings
    U_GAIN = 1000
    P_TOL = np.array([2, 1000, 1000, 1000, 0.02, 0.02, 0.02])
    D_MAX = np.array([
        0,0,0, 0,0,0, 4,4,4, 4,4,4,
    ])
    D_MIN = -D_MAX

    # sim settings
    SIM_T = CTRL_T / 10

    # file access
    xfilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_xref.npy"
    ufilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_uref.npy"
    dfilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_d.npy"
    trajname = "/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_smlms_traj.npy"
    figname = "/home/derek/dev/my-repos/qrac/experiments/urop/figures/setpt_smlms.png"


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

    # realistic parameter bounds
    p_min = np.zeros(7)
    p_min[0] = 1 / (2*acc.m)
    p_min[1] = 1 / (10*acc.Ixx)
    p_min[2] = 1 / (10*acc.Iyy)
    p_min[3] = 1 / (10*acc.Izz)
    p_min[4] = (acc.Izz - 10*acc.Iyy) / acc.Ixx
    p_min[5] = (acc.Ixx - 10*acc.Izz) / acc.Iyy
    p_min[6] = (acc.Iyy - 10*acc.Ixx) / acc.Izz

    p_max = np.zeros(7)
    p_max[0] = 1 / acc.m
    p_max[1] = 1 / (acc.Ixx)
    p_max[2] = 1 / (acc.Iyy)
    p_max[3] = 1 / (acc.Iyy)
    p_max[4] = (10*acc.Izz - acc.Iyy) / acc.Ixx
    p_max[5] = (10*acc.Ixx - acc.Izz) / acc.Iyy
    p_max[6] = (10*acc.Iyy - acc.Ixx) / acc.Izz



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
        qp_tol=10**-6, max_iter=10
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

    anmpc.start()
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
    anmpc.stop()

    print(f"acc params:\n{AffineQuadrotor(acc).get_parameters()}")
    print(f"inacc params:\n{AffineQuadrotor(inacc).get_parameters()}")
    print(f"param min: \n{p_min}")
    print(f"param max: \n{p_max}")


    # save trajectory data
    xdata = sim.get_xdata()
    np.save(trajname, xdata)


    # calculate RMSE
    err = np.sum(np.abs( xref[:,0:3] - xdata[:,0:3] )**2,axis=-1)**0.5
    rmse = np.sqrt(np.sum(np.square(err))/err.shape[0])
    print(f"RMSE: {rmse}")


    # save plot
    sim.get_plot(figname)
    #sim.get_animation("/home/derek/dev/my-repos/qrac/experiments/urop/figures/setpt_anim.gif")


if __name__=="__main__":
    main()

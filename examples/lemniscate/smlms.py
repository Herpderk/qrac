#!/usr/bin/python3

import numpy as np
from qrac.models import Crazyflie, Quadrotor, AffineQuadrotor
from qrac.control import AdaptiveNMPC
from qrac.estimation import SetMembershipEstimator, LMS
from qrac.sim import MinimalSim


def run():
    # mpc settings
    CTRL_T = 0.01
    NODES = 100
    Q = np.diag([10,10,10, 1,1,1,1, 1,1,1, 1,1,1,])
    R = np.diag([0, 0, 0, 0])

    # estimator settings
    U_GAIN = 1000
    P_TOL = 0.1*np.ones(10)
    D_MAX = np.array([
        0,0,0, 0,0,0,0, 5,5,5, 5,5,5,
    ])
    D_MIN = -D_MAX

    # sim settings
    SIM_T = CTRL_T / 10

    # file access
    xfilename = "../refs/lemniscate/xref.npy"
    ufilename = "../refs/lemniscate/uref.npy"

    # load in time optimal trajectory
    xref = np.load(xfilename)
    uref = np.load(ufilename)


    # inaccurate model
    inacc = Crazyflie(Ax=0, Ay=0, Az=0)

    # true model
    m_true = 1.5 * inacc.m
    Ixx_true = 2 * inacc.Ixx
    Iyy_true = 2 * inacc.Iyy
    Izz_true = 2 * inacc.Izz
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

    # init set-membership
    p_min = AffineQuadrotor(inacc).get_parameters()\
        - 1*np.abs(AffineQuadrotor(inacc).get_parameters())
    p_max = AffineQuadrotor(inacc).get_parameters()\
        + 1*np.abs(AffineQuadrotor(inacc).get_parameters())

    # init LMS
    lms = LMS(
        model=inacc, param_min=p_min, param_max=p_max,
        update_gain=U_GAIN, time_step=CTRL_T
    )
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
        nlp_tol=10**-6, nlp_max_iter=1, qp_max_iter=4
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
        d = D_MAX * np.random.uniform(-1, 1, nx)
        x = sim.update(x=x, u=u, d=d, timer=True)

        print(f"\nu: {u}")
        print(f"x: {x}")
        print(f"sim time: {(k+1)*CTRL_T}\n")

    print(f"acc params:\n{AffineQuadrotor(acc).get_parameters()}")
    print(f"inacc params:\n{AffineQuadrotor(inacc).get_parameters()}")
    print(f"param min: \n{p_min}")
    print(f"param max: \n{p_max}")


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
    sim.get_animation()


if __name__=="__main__":
    run()

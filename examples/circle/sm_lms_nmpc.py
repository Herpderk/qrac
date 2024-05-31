#!/usr/bin/python3

import numpy as np
from qrac.models import Crazyflie, Quadrotor, AffineQuadrotor
from qrac.control import AdaptiveNMPC
from qrac.estimation import SetMembershipEstimator, LMS
from qrac.sim import MinimalSim


def run():
    # mpc settings
    CTRL_T = 0.005
    NODES = 40
    Q = np.diag([1,1,1, 1,1,1,1, 1,1,1, 1,1,1,])
    R = np.diag([0, 0, 0, 0])

    # estimator settings
    U_GAIN = 400
    P_TOL = 0.2*np.ones(7)
    D_MAX = np.array([
        0,0,0, 0,0,0,0, 6,6,6, 6,6,6,
    ])
    D_MIN = -D_MAX

    # sim settings
    SIM_T = CTRL_T / 10

    xfilename = "/home/derek/dev/my-repos/qrac/examples/refs/circle/xref.npy"
    ufilename = "/home/derek/dev/my-repos/qrac/examples/refs/circle/uref.npy"


    # load in time optimal trajectory
    xref = np.load(xfilename)
    uref = np.load(ufilename)


    # inaccurate model
    inacc = Crazyflie(Ax=0, Ay=0, Az=0)

   # true model
    m_true = 2 * inacc.m
    Ixx_true = 4 * inacc.Ixx
    Iyy_true = 4 * inacc.Iyy
    Izz_true = 1 * inacc.Izz
    Ax_true = 0
    Ay_true = 0
    Az_true = 0
    xB_true = inacc.xB
    yB_true = inacc.yB
    kf_true = inacc.kf
    km_true = inacc.km
    u_min_true = inacc.u_min
    u_max_true = inacc.u_max
    acc = Quadrotor(
        m_true, Ixx_true, Iyy_true, Izz_true,
        Ax_true, Ay_true, Az_true, kf_true, km_true,
        xB_true, yB_true, u_min_true, u_max_true)


    # define realistic parameter bounds
    p_min = np.zeros(7)
    p_min[0] = 1 / (2.5*inacc.m)
    #p_min[1:4] = np.zeros(3)
    # moments of inertia bounds based on max moment of x5
    p_min[1] = 1 / (5*inacc.Ixx)
    p_min[2] = 1 / (5*inacc.Iyy)
    p_min[3] = 1 / (5*inacc.Izz)
    p_min[4] = (inacc.Izz - 5*inacc.Iyy) / inacc.Ixx
    p_min[5] = (inacc.Ixx - 2*inacc.Izz) / inacc.Iyy
    p_min[6] = (inacc.Iyy - 5*inacc.Ixx) / inacc.Izz

    p_max = np.zeros(7)
    p_max[0] = 1 / inacc.m
    #p_max[1:4] = np.zeros(3)
    p_max[1] = 1 / (inacc.Ixx)
    p_max[2] = 1 / (inacc.Iyy)
    p_max[3] = 1 / (inacc.Iyy)
    p_max[4] = (2*inacc.Izz - inacc.Iyy) / inacc.Ixx
    p_max[5] = (5*inacc.Ixx - inacc.Izz) / inacc.Iyy
    p_max[6] = (5*inacc.Iyy - inacc.Ixx) / inacc.Izz

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
        num_nodes=NODES, rti=True, real_time=True,
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
        d = 2*D_MAX*(-0.5 + np.random.rand(13))
        x = sim.update(x=x, u=u, d=d, timer=True)

        print(f"\nu: {u}")
        print(f"x: {x}")
        print(f"sim time: {(k+1)*CTRL_T}\n")
    anmpc.stop()

    print(f"init param min: \n{p_min}")
    print(f"init param max: \n{p_max}")
    print(f"acc params:\n{AffineQuadrotor(acc).get_parameters()}")
    print(f"inacc params:\n{AffineQuadrotor(inacc).get_parameters()}")
    
    sim.get_animation()



if __name__=="__main__":
    run()

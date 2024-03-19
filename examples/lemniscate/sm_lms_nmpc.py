#!/usr/bin/python3

from qrac.models import Crazyflie, Quadrotor, AffineQuadrotor
from qrac.trajectory import LemniScate
from qrac.control import AdaptiveNMPC
from qrac.estimation import SetMembershipEstimator, LMS
from qrac.sim import MinimalSim
import numpy as np


def run():
    # mpc settings
    CTRL_T = 0.01
    NODES = 150
    Q = np.diag([1,1,1, 1,1,1, 1,1,1, 1,1,1,])
    R = np.diag([0, 0, 0, 0])

    # estimator settings
    U_GAIN = 1000
    P_TOL = 0.1*np.ones(10)
    D_MAX = np.array([
        0,0,0, 0,0,0, 10,10,10, 10,10,10,
    ])
    D_MIN = -D_MAX

    # sim settings
    SIM_T = CTRL_T / 10


    # load in time optimal trajectory
    xref = np.load("refs/xref.npy")
    uref = np.load("refs/uref.npy")


    # inaccurate model
    inacc = Crazyflie(Ax=0, Ay=0, Az=0)

   # true model
    m_true = 1.5 * inacc.m
    Ixx_true = 1.8 * inacc.Ixx
    Iyy_true = 1.8 * inacc.Iyy
    Izz_true = 1.8 * inacc.Izz
    Ax_true = 0
    Ay_true = 0
    Az_true = 0
    xB_true = inacc.xB
    yB_true = inacc.yB
    k_true = inacc.k
    u_min_true = inacc.u_min
    u_max_true = inacc.u_max
    acc = Quadrotor(
        m_true, Ixx_true, Iyy_true, Izz_true,
        Ax_true, Ay_true, Az_true, xB_true, yB_true,
        k_true, u_min_true, u_max_true)


    # init LMS
    lms = LMS(
        model=inacc, update_gain=U_GAIN, time_step=CTRL_T
    )

    # init set-membership
    p_min = AffineQuadrotor(acc).get_parameters()\
        - 2*np.abs(AffineQuadrotor(acc).get_parameters())
    p_max = AffineQuadrotor(acc).get_parameters()\
        + 2*np.abs(AffineQuadrotor(acc).get_parameters())
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
        d = 2*D_MAX*(-0.5 + np.random.rand(12))
        x = sim.update(x=x, u=u, d=d, timer=True)

        print(f"\nu: {u}")
        print(f"x: {x}")
        print(f"sim time: {(k+1)*CTRL_T}\n")
    anmpc.stop()
    
    print(f"acc params:\n{AffineQuadrotor(acc).get_parameters()}")
    print(f"inacc params:\n{AffineQuadrotor(inacc).get_parameters()}")

    # plot
    sim.get_animation(
        filename="/home/derek/Documents/qrac/smlms_exp.gif"
    )


if __name__=="__main__":
    run()

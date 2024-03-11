#!/usr/bin/python3

from qrac.models import Crazyflie, Quadrotor, ParameterizedQuadrotor
from qrac.trajectory import Circle
from qrac.control.param_nmpc import ParameterAdaptiveNMPC
from qrac.estimation import MHE
from qrac.sim import MinimalSim
import numpy as np


def run():
    # mpc settings
    CTRL_T = 0.005
    NODES = 50
    Q = np.diag([1,1,1, 2,2,2, 1,1,1, 2,2,2,])
    R = np.diag([0, 0, 0, 0])

    # estimator settings
    Q_MHE = 10**-3 * np.diag([1, 1, 1, 1, 1, 1, 1])
    R_MHE = 10**3 * np.diag([1,1,1, 1,1,1, 1,1,1, 1,1,1])
    NODES_MHE = 20
    D_MIN = -0.1*np.ones(12)
    D_MAX = -D_MIN

    # sim settings
    SIM_TIME = 30
    SIM_T = CTRL_T / 10


    # inaccurate model
    inacc = Crazyflie(Ax=0, Ay=0, Az=0)

    # true plant model
    m_true = 1.5 * inacc.m
    Ixx_true = 50 * inacc.Ixx
    Iyy_true = 50 * inacc.Iyy
    Izz_true = 50 * inacc.Izz
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


    # init estimator
    p_min = np.zeros(7)
    p_max = 3 * ParameterizedQuadrotor(acc).get_parameters()
    mhe = MHE(
        model=inacc, Q=Q_MHE, R=R_MHE,
        param_min=p_min, param_max=p_max,
        disturb_min=D_MIN, disturb_max=D_MAX,
        time_step=CTRL_T, num_nodes=NODES_MHE,
        rti=True, nonlinear=True,
        nlp_tol=10**-6, nlp_max_iter=1, qp_max_iter=3,
    )

    # init mpc
    adaptive_mpc = ParameterAdaptiveNMPC(
        model=inacc, estimator=mhe, Q=Q, R=R,
        u_min=inacc.u_min, u_max=inacc.u_max, time_step=CTRL_T,
        num_nodes=NODES, real_time=False, rti=True,
        nlp_tol=10**-6, nlp_max_iter=1, qp_max_iter=5
    )


    # init sim
    nx = inacc.nx
    x0 = np.zeros(nx)
    x0[0] = 8
    steps = int(round(SIM_TIME / CTRL_T))
    sim = MinimalSim(
        x0=x0, model=acc,
        sim_step=SIM_T, control_step=CTRL_T,
        data_len=steps
    )


    # define a circular trajectory
    traj = Circle(v=4, r=8, alt=8)


    # run for predefined number of steps
    x = x0
    xset = np.zeros(nx*NODES)

    for k in range(steps):
        t = k*CTRL_T
        for n in range(NODES):
            xset[n*nx : n*nx + nx] = traj.get_setpoint(t)
            t += CTRL_T
        u = adaptive_mpc.get_input(x=x, xset=xset, timer=True)
        x = sim.update(x=x, u=u, timer=True)

        print(f"\nu: {u}")
        print(f"x: {x}")
        print(f"sim time: {(k+1)*CTRL_T}\n")
    
    print("acc params:")
    print(ParameterizedQuadrotor(acc).get_parameters())
    print("inacc params:")
    print(ParameterizedQuadrotor(inacc).get_parameters())
    xdata = sim.get_xdata()


if __name__=="__main__":
    run()


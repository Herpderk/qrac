#!/usr/bin/python3

import numpy as np
from qrac.dynamics import NonlinearQuadrotor, NonlinearCrazyflie
from qrac.control.acados_mpc import AcadosMpc
from qrac.control.l1_augment import L1Augmentation
from qrac.sim.acados_plant import AcadosPlant
from qrac.sim.minimal_sim import MinimalSim


def main():
    # inaccurate model
    model_inacc = NonlinearCrazyflie(Ax=0, Ay=0, Az=0)

    # true plant model
    m_true = 8 * model_inacc.m
    Ixx_true = 8 * model_inacc.Ixx
    Iyy_true = 8 * model_inacc.Iyy
    Izz_true = 8 * model_inacc.Izz
    Ax_true = 0
    Ay_true = 0
    Az_true = 0
    bx_true = model_inacc.bx
    by_true = model_inacc.by
    k_true = model_inacc.k
    model_acc = NonlinearQuadrotor(
        m_true, Ixx_true, Iyy_true, Izz_true,
        Ax_true, Ay_true, Az_true, bx_true, by_true, k_true)

    # initialize mpc
    Q = np.diag([10,10,10, 1,1,1, 2,2,2, 1,1,1,])
    R = 0.1 * np.diag([1, 1, 1, 1])
    max_thrust = 0.64
    u_max = max_thrust * np.ones(4)
    u_min = np.zeros(4)
    mpc_T = 0.02
    N = 16
    rt = False
    mpc = AcadosMpc(
        model=model_inacc, Q=Q, R=R, u_max=u_max, u_min=u_min,
        time_step=mpc_T, num_nodes=N, real_time=rt)

    # initialize L1 augmentation
    As = 0.05 * np.array([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1],
    ])
    w = 0.2
    l1_T = mpc_T/5
    l1_mpc = L1Augmentation(
        model=model_inacc, control_ref=mpc, adapt_gain=As,
        cutoff_freq=w, time_step=l1_T, real_time=rt)

    # initialize simulator plant
    sim_T = l1_T/100
    plant = AcadosPlant(
        model=model_acc, sim_step=sim_T, control_step=l1_T)

    # initialize the simulator
    lb_pose = [-10, -10, 0]
    ub_pose = [10, 10, 10]
    sim = MinimalSim(
        plant=plant, controller=l1_mpc, lb_pose=lb_pose, ub_pose=ub_pose)

    # define the initial state and setpoint
    x0 = np.zeros(12)
    x_set = np.array([5, 5, 5, 0,0,0, 0,0,0, 0,0,0])

    # run the sim for N control loops
    N = int(round(20 / l1_T))
    sim.start(x0=x0, max_steps=N, verbose=True)
    sim.update_setpoint(x_set=x_set)


if __name__=="__main__":
    main()

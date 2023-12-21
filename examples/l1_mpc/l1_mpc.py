#!/usr/bin/python3

import numpy as np
from qrac.dynamics import Quadrotor, Crazyflie
from qrac.trajectory import Circle
from qrac.control.acados_mpc import AcadosMpc
from qrac.control.l1_augment import L1Augmentation
from qrac.sim.acados_plant import AcadosPlant
from qrac.sim.minimal_sim import MinimalSim


def main():
    # inaccurate model
    model_inacc = Crazyflie(Ax=0, Ay=0, Az=0)

    # true plant model
    m_true = 4 * model_inacc.m
    Ixx_true = 4 * model_inacc.Ixx
    Iyy_true = 4 * model_inacc.Iyy
    Izz_true = 4 * model_inacc.Izz
    Ax_true = 0
    Ay_true = 0
    Az_true = 0
    bx_true = model_inacc.bx
    by_true = model_inacc.by
    k_true = model_inacc.k
    model_acc = Quadrotor(
        m_true, Ixx_true, Iyy_true, Izz_true,
        Ax_true, Ay_true, Az_true, bx_true, by_true, k_true)

    # initialize mpc
    Q = np.diag([8,8,8, 0.4,0.4,0.4, 2,2,2, 0.4,0.4,0.4,])
    R = np.diag([0, 0, 0, 0])
    max_thrust = 0.64
    u_max = max_thrust * np.ones(4)
    u_min = np.zeros(4)
    mpc_T = 0.03
    num_nodes = 10
    rt = False
    mpc = AcadosMpc(
        model=model_inacc, Q=Q, R=R, u_max=u_max, u_min=u_min,
        time_step=mpc_T, num_nodes=num_nodes, real_time=rt)

    # initialize L1 augmentation
    a_gain = 600
    w = 60
    l1_T = mpc_T
    l1_mpc = L1Augmentation(
        model=model_inacc, control_ref=mpc, adapt_gain=a_gain,
        bandwidth=w, time_step=l1_T, real_time=rt)

    # initialize simulator plant
    sim_T = l1_T/100
    plant = AcadosPlant(
        model=model_acc, sim_step=sim_T, control_step=l1_T)

    # initialize the simulator
    lb_pose = [-10, -10, 0]
    ub_pose = [10, 10, 10]
    sim = MinimalSim(
        plant=plant, controller=l1_mpc, lb_pose=lb_pose, ub_pose=ub_pose, data_len=2000)

    # define a circular trajectory
    traj = Circle(r=4, alt=4)

    # Run the sim for N control loops
    x0 = np.zeros(model_inacc.nx)
    N = int(round(20 / l1_T))      # 20 seconds worth of control loops
    sim.start(x0=x0, max_steps=N, verbose=True)

    # track the given trajectory
    x_set = np.zeros(mpc.n_set)
    nx = model_inacc.nx
    dt = mpc.dt
    t0 = sim.timestamp
    while sim.is_alive:
        t = sim.timestamp
        tk = sim.timestamp
        for k in range(num_nodes):
            x_set[k*nx : k*nx + nx] = \
                np.array(traj.get_setpoint(tk - t0))
            tk += dt
        sim.update_setpoint(x_set=x_set)
        while sim.timestamp < t+dt and sim.is_alive:
            pass


if __name__=="__main__":
    main()

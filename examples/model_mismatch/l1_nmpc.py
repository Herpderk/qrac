#!/usr/bin/python3

import numpy as np
from qrac.models import Quadrotor, Crazyflie
from qrac.trajectory import Circle
from qrac.control.nmpc import NMPC
from qrac.control.l1_aug import L1Augmentation
from qrac.sim.acados_plant import AcadosPlant
from qrac.sim.minimal_sim import MinimalSim


def main():
    # inaccurate model
    model_inacc = Crazyflie(Ax=0, Ay=0, Az=0)

    # true plant model
    m_true = 1.5 * model_inacc.m
    Ixx_true = 60 * model_inacc.Ixx
    Iyy_true = 60 * model_inacc.Iyy
    Izz_true = 60 * model_inacc.Izz
    Ax_true = 0
    Ay_true = 0
    Az_true = 0
    xB_true = model_inacc.xB
    yB_true = model_inacc.yB
    k_true = model_inacc.k
    u_min_true = model_inacc.u_min
    u_max_true = model_inacc.u_max
    model_acc = Quadrotor(
        m_true, Ixx_true, Iyy_true, Izz_true,
        Ax_true, Ay_true, Az_true, xB_true, yB_true, k_true, u_min_true, u_max_true)

    # initialize mpc
    Q = np.diag([40,40,40, 10,10,10, 20,20,20, 10,10,10])
    R = np.diag([0, 0, 0, 0])
    u_min = model_inacc.u_min
    u_max = model_inacc.u_max
    mpc_T = 0.006
    num_nodes = 100
    rti = True
    real_time = False
    mpc = NMPC(
        model=model_inacc, Q=Q, R=R, u_max=u_max, u_min=u_min,
        time_step=mpc_T, num_nodes=num_nodes, rti=rti,)

    # initialize L1 augmentation
    a_gain = 50
    w = 10
    l1_T = mpc_T / 1
    l1_mpc = L1Augmentation(
        model=model_inacc, control_ref=mpc, adapt_gain=a_gain,
        bandwidth=w, time_step=l1_T, real_time=real_time)

    # initialize simulator plant
    sim_T = l1_T / 10
    plant = AcadosPlant(
        model=model_acc, sim_step=sim_T, control_step=l1_T)

    # initialize the simulator
    lb_pose = [-10, -10, 0]
    ub_pose = [10, 10, 10]
    sim = MinimalSim(
        plant=plant, controller=l1_mpc, lb_pose=lb_pose, ub_pose=ub_pose, data_len=2000)

     # define a circular trajectory
    traj = Circle(v=4, r=8, alt=8)

    # Run the sim for N control loops
    x0 = np.array([8,0,0, 0,0,0, 0,0,0, 0,0,0])
    N = int(round(30 / l1_T))      # 30 seconds worth of control loops
    sim.start(x0=x0, max_steps=N, verbose=True)

    # track the given trajectory
    xset = np.zeros(mpc.n_set)
    nx = model_inacc.nx
    dt = mpc.dt
    t0 = sim.timestamp
    while sim.is_alive:
        t = sim.timestamp
        for k in range(num_nodes):
            xset[k*nx : k*nx + nx] = \
                np.array(traj.get_setpoint(t - t0))
            t += dt
        sim.update_setpoint(xset=xset)


if __name__=="__main__":
    main()

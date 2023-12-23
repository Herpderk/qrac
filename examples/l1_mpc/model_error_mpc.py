#!/usr/bin/python3

from qrac.dynamics import Crazyflie, Quadrotor
from qrac.trajectory import Circle
from qrac.control.acados_mpc import AcadosMpc
from qrac.sim.acados_plant import AcadosPlant
from qrac.sim.minimal_sim import MinimalSim
import numpy as np


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

    # initialize controller
    Q = np.diag([40,40,40, 1,1,1, 20,20,20, 1,1,1])
    R = np.diag([0, 0, 0, 0])
    max_thrust = 0.64           # N
    u_max = max_thrust * np.ones(4)
    u_min = np.zeros(4)
    mpc_T = 0.02
    num_nodes = 14
    rt = False
    mpc = AcadosMpc(
        model=model_inacc, Q=Q, R=R, u_max=u_max, u_min=u_min, \
        time_step=mpc_T, num_nodes=num_nodes, real_time=rt,)

    # initialize simulator plant
    sim_T = mpc_T / 100
    plant = AcadosPlant(
        model=model_acc, sim_step=sim_T, control_step=mpc_T)

    # initialize simulator
    lb_pose = [-10, -10, 0]
    ub_pose = [10, 10, 10]
    sim = MinimalSim(
        plant=plant, controller=mpc, lb_pose=lb_pose, ub_pose=ub_pose,)

     # define a circular trajectory
    traj = Circle(v=8, r=4, alt=4)

    # Run the sim for N control loops
    x0 = np.array([4,0,0, 0,0,0, 0,0,0, 0,0,0])
    N = int(round(20 / mpc_T))      # 20 seconds worth of control loops
    sim.start(x0=x0, max_steps=N, verbose=True)

    # track the given trajectory
    x_set = np.zeros(mpc.n_set)
    nx = model_inacc.nx
    dt = mpc.dt
    t0 = sim.timestamp
    while sim.is_alive:
        t = sim.timestamp
        for k in range(num_nodes):
            x_set[k*nx : k*nx + nx] = \
                np.array(traj.get_setpoint(t - t0))
            t += dt
        sim.update_setpoint(x_set=x_set)


if __name__=="__main__":
    main()
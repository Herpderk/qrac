#!/usr/bin/python3

from qrac.dynamics import Crazyflie, Quadrotor, ParameterAffineQuadrotor
from qrac.trajectory import Circle
from qrac.control.sm_nmpc import SetMembershipMPC
from qrac.sim.acados_plant import AcadosPlant
from qrac.sim.minimal_sim import MinimalSim
import numpy as np


def main():
    # inaccurate model
    model_inacc = Crazyflie(Ax=0, Ay=0, Az=0)

    # true plant model
    m_true = 2 * model_inacc.m
    Ixx_true = 25 * model_inacc.Ixx
    Iyy_true = 25 * model_inacc.Iyy
    Izz_true = 25 * model_inacc.Izz
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
        Ax_true, Ay_true, Az_true, xB_true, yB_true,
        k_true, u_min_true, u_max_true)

    # initialize mpc
    Q = np.diag([40,40,40, 10,10,10, 20,20,20, 10,10,10])
    R = np.diag([0, 0, 0, 0])
    update_gain = 200
    param_min = ParameterAffineQuadrotor(model_acc).get_parameters()\
        - 2*np.abs(ParameterAffineQuadrotor(model_acc).get_parameters())
    param_max = ParameterAffineQuadrotor(model_acc).get_parameters()\
        + 2*np.abs(ParameterAffineQuadrotor(model_acc).get_parameters())
    disturb_max = 0.001*np.ones(12)
    disturb_min = -disturb_max
    u_max = u_max_true #model_inacc.u_max
    u_min = u_min_true
    mpc_T = 0.04
    num_nodes = 50
    rt = False
    mpc = SetMembershipMPC(
        model=model_inacc, Q=Q, R=R, update_gain=update_gain,
        param_min=param_min,param_max=param_max,
        disturb_max=disturb_max, disturb_min=disturb_min,
        u_min=u_min, u_max=u_max,
        time_step=mpc_T, num_nodes=num_nodes, real_time=rt,)

    # initialize simulator plant
    sim_T = mpc_T / 10
    plant = AcadosPlant(
        model=model_acc, sim_step=sim_T, control_step=mpc_T)

    # initialize simulator
    lb_pose = [-10, -10, 0]
    ub_pose = [10, 10, 10]
    sim = MinimalSim(
        plant=plant, controller=mpc, lb_pose=lb_pose, ub_pose=ub_pose,)

     # define a circular trajectory
    traj = Circle(v=4, r=8, alt=8)

    # Run the sim for N control loops
    x0 = np.array([8,0,0, 0,0,0, 0,0,0, 0,0,0])
    N = int(round(30 / mpc_T))      # 30 seconds worth of control loops
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
    
    print("inacc params:")
    print(
        ParameterAffineQuadrotor(model_inacc).get_parameters()
    )
    print("acc params:")
    print(  
        ParameterAffineQuadrotor(model_acc).get_parameters()
    )
    print(param_min)
    print(param_max)


if __name__=="__main__":
    main()
#!/usr/bin/python3

from aCBF_QP_MPC.dynamics.acados import nonlinear_quadrotor_model
from aCBF_QP_MPC.control.acados import SqpNmpc
from aCBF_QP_MPC.sim.acados import SimBackend, SimFrontend

import numpy as np
import time

# crazyflie system identification
# https://www.research-collection.ethz.ch/handle/20.500.11850/214143


def main():
    # SI Units
    m = 0.028                   # kg
    l = 0.040                   # m
    Ixx = 3.144988 * 10**(-5)
    Iyy = 3.151127 * 10**(-5)
    Izz = 7.058874 * 10**(-5)
    k = 0.005964552             # k is the ratio of torque to thrust
    Ax = 0
    Ay = 0
    Az = 0
    max_thrust = 0.64           # N


    T = 0.02
    N = 8
    Q = np.diag(
        [4,4,4, 2.5,2.5,2.5, 1,1,1, 1,1,1,])
    R = np.diag(
        [1, 1, 1, 1])
    u_max = max_thrust * np.ones(4)
    u_min = np.zeros(4)


    x0 = np.zeros(12)
    x_set = np.array(
        [5,-4,10, 0,0,0, 0,0,0, 0,0,0])


    model = nonlinear_quadrotor_model(
        m=m, l=l, Ixx=Ixx, Iyy=Iyy, Izz=Izz, k=k)
    test_visual = SimFrontend(x0=x0,)
    test_sim = SimBackend(
        model=model, time_step=T)
    test_mpc = SqpNmpc(
        model=model, Q=Q, R=R, u_max=u_max, u_min=u_min, time_step=T, num_nodes=N,)


    try:
        x = x0
        while True:
            st = time.perf_counter()
            u = test_mpc.get_next_control(x0=x, x_set=x_set, timer=True)
            x = test_sim.run_control_loop(x0=x, u=u, timer=True)
            test_visual.update_pose(x)
            print(f"u: {u}")
            print(f"x: {x}")
            
            while time.perf_counter() - st <= T:
                pass
            dt = time.perf_counter() - st
    except KeyboardInterrupt:
        pass


if __name__=="__main__":
    main()
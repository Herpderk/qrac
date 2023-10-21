#!/usr/bin/python3

from aCBF_QP_MPC.dynamics.acados_model import get_nonlinear_acados_model
from aCBF_QP_MPC.control.acados_mpc import AcadosMpc
import numpy as np


# crazyflie system identification
# https://www.research-collection.ethz.ch/handle/20.500.11850/214143


def main():
    # Initialize model (SI units)
    m = 0.028                   # kg
    l = 0.040                   # m
    Ixx = 3.144988 * 10**(-5)
    Iyy = 3.151127 * 10**(-5)
    Izz = 7.058874 * 10**(-5)
    k = 0.005964552             # k is the ratio of torque to thrust
    Ax = 0
    Ay = 0
    Az = 0
    model = get_nonlinear_acados_model(
        m=m, l=l, Ixx=Ixx, Iyy=Iyy, Izz=Izz, k=k)


    # initialize controller
    Q = np.diag([10,10,10, 1,1,1, 2.8,2.8,2.8, 1,1,1,])
    R = 0.1 * np.diag([1, 1, 1, 1])
    max_thrust = 0.64   # N
    u_max = max_thrust * np.ones(4)
    u_min = np.zeros(4)
    control_T = 0.025
    N = 8
    mpc = AcadosMpc(
        model=model, Q=Q, R=R, u_max=u_max, u_min=u_min, time_step=control_T, num_nodes=N,)


    # run a single control loop with trajectory prediction
    x0 = np.zeros(12)
    x_set = np.array([5,-4,10, 0,0,0, 0,0,0, 0,0,0])
    mpc.get_next_state(x0, x_set, timer=True, visuals=True)


if __name__=="__main__":
    main()
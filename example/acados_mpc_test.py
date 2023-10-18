#!/usr/bin/python3

from aCBF_QP_MPC.dynamics.casadi import nonlinear_quadrotor_model
from aCBF_QP_MPC.control.acados import sqpNMPC
import numpy as np

# crazyflie system identification
# https://www.research-collection.ethz.ch/handle/20.500.11850/214143


def main():
    m = 0.028 # kg
    l = 0.040 # m
    Ixx = 3.144988 * 10**(-5)
    Iyy = 3.151127 * 10**(-5)
    Izz = 7.058874 * 10**(-5)
    k = 0.005964552          # k is the ratio of torque to thrust
    Ax = 0
    Ay = 0
    Az = 0
    max_thrust = 0.64
    '''
    kf = 1.858 * 10**(-3)#0.005022
    km = 1.858 * 10**(-5)
    '''
    model = nonlinear_quadrotor_model(
        m=m, l=l, Ixx=Ixx, Iyy=Iyy, Izz=Izz, k=k)

    Q = np.diag(
        [1,1,1, 1,1,1, 1,1,1, 1,1,1,])
    R = np.diag(
        [1, 1, 1, 1])
    u_max = max_thrust * np.ones(4)
    u_min = np.zeros(4)
    test_mpc = sqpNMPC(
        model=model, Q=Q, R=R, u_max=u_max, u_min=u_min, time_step=0.05, num_nodes=25,)

    x0 = np.zeros(12)
    x_set = np.array(
        [-5,5,10, 0,0,0, 0,0,0, 0,0,0])
    test_mpc.get_next_state(
        x0=x0, x_set=x_set, timer=True, visuals=True)
    
    while True:
        test_mpc.get_next_control(
            x0=x0, x_set=x_set, timer=True,)

    
if __name__=="__main__":
    main()
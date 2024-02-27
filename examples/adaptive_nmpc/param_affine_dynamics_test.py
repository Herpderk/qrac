#!/usr/bin/python3

from qrac.dynamics import Crazyflie, ParamAffineQuadrotor
from qrac.control.nmpc import NMPC
import casadi as cs
import numpy as np


def main():
    model = Crazyflie(Ax=0, Ay=0, Az=0)

    Q = np.diag([1,1,1, 1,1,1, 1,1,1, 1,1,1])
    R = np.diag([0, 0, 0, 0])
    u_max = model.u_max
    u_min = np.zeros(4)
    control_T = 0.05
    num_nodes = 200
    rt = False
    mpc = NMPC(
        model=model, Q=Q, R=R, u_max=u_max, u_min=u_min,
        time_step=control_T, num_nodes=num_nodes, real_time=rt,)

    x0 = np.zeros(12)
    x_set = np.tile(
        np.array([2,-2, 4, 0,0,0, 0,0,0, 0,0,0]), num_nodes)
    xs, us = mpc.get_trajectory(x0, x_set, timer=True, visuals=True)


    # Let's confirm that the same sequence of control inputs will yield
    # the same trajectory in the augmented dynamics
    model = ParamAffineQuadrotor(model)
    param = np.array([
        1/model.m , 0, 0, 0,
        1/model.Ixx, 1/model.Iyy, 1/model.Izz,
        (model.Izz-model.Iyy)/model.Ixx,
        (model.Ixx-model.Izz)/model.Iyy,
        (model.Iyy-model.Ixx)/model.Izz
    ])
    
    f = cs.Function("f", [model.x, model.u], [model.xdot])
    def rk4(f, dt, x, u):
        k1 = f(x,u)
        k2 = f(x + dt/2 * k1, u)
        k3 = f(x + dt/2 * k2, u)
        k4 = f(x + dt * k3,  u)
        xf = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return np.array(xf).flatten()

    xs = np.zeros((num_nodes, model.x.shape[0]))
    xs[0,:] = np.concatenate((x0, param))
    for k in range(1, num_nodes):
        xs[k,:] = rk4(f, control_T, xs[k-1,:], us[k-1,:])
    mpc._vis_plots(xs, us)


if __name__=="__main__":
    main()
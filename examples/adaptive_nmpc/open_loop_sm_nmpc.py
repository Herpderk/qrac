#!/usr/bin/python3

from qrac.dynamics import Crazyflie
from qrac.control.sm_nmpc import SetMembershipMPC
import numpy as np


def main():
    # get dynamics model and convert to acados
    model = Crazyflie(Ax=0, Ay=0, Az=0)

    # initialize controller
    Q = np.diag([1,1,1, 1,1,1, 1,1,1, 1,1,1])
    R = np.diag([0, 0, 0, 0])
    u_max = model.u_max
    u_min = np.zeros(4)
    control_T = 0.05
    num_nodes = 200
    rt = False
    param_max = np.array([
        50, 0, 0, 0,
        50000, 50000, 20000,
        5, 5, 0
    ])
    param_min = np.array([
        0, 0, 0, 0,
        0, 0, 0,
        -5, -5, 0
    ])
    disturb_max = 1*np.ones(12)
    disturb_min = np.zeros(12)
    mpc = SetMembershipMPC(
        model=model, Q=Q, R=R, update_gain=0.1,
        param_min=param_min,param_max=param_max,
        disturb_max=disturb_max, disturb_min=disturb_min,
        u_max=u_max, u_min=u_min,
        time_step=control_T, num_nodes=num_nodes, real_time=rt,)

    # run a single control loop with trajectory prediction
    x0 = np.zeros(12)
    # the setpoint must span the entire prediction horizon
    x_set = np.tile(
        np.array([-2, 2, 4, 0,0,0, 0,0,0, 0,0,0]), num_nodes)
    mpc.get_trajectory(x0, x_set, timer=True, visuals=True)


if __name__=="__main__":
    main()
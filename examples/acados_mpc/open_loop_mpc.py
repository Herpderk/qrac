#!/usr/bin/python3

from qrac.dynamics import Crazyflie
from qrac.control.acados_mpc import AcadosMpc
import numpy as np


def main():
    # get dynamics model and convert to acados
    model = Crazyflie(Ax=0, Ay=0, Az=0)

    # initialize controller
    Q = np.diag([1,1,1, 1,1,1, 1,1,1, 1,1,1])
    R = np.diag([0, 0, 0, 0])
    max_thrust = 0.64           # N
    u_max = max_thrust * np.ones(4)
    u_min = np.zeros(4)
    control_T = 0.01
    num_nodes = 400
    rt = False
    mpc = AcadosMpc(
        model=model, Q=Q, R=R, u_max=u_max, u_min=u_min, \
        time_step=control_T, num_nodes=num_nodes, real_time=rt,)

    # run a single control loop with trajectory prediction
    x0 = np.zeros(12)
    # the setpoint must span the entire prediction horizon
    x_set = np.tile(
        np.array([2,-2, 4, 0,0,0, 0,0,0, 0,0,0]), num_nodes)
    mpc.get_state(x0, x_set, timer=True, visuals=True)


if __name__=="__main__":
    main()
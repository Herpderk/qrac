#!/usr/bin/python3

from qrac.models import Crazyflie
from qrac.control import NMPC
import numpy as np


def main():
    # get dynamics model and convert to acados
    model = Crazyflie(Ax=0, Ay=0, Az=0)

    # initialize controller
    Q = np.diag([1,1,1, 1,1,1,1, 1,1,1, 1,1,1])
    R = np.diag([0, 0, 0, 0])
    u_min = model.u_min
    u_max = model.u_max
    control_T = 0.05
    num_nodes = 200
    rti = False
    mpc = NMPC(
        model=model, Q=Q, R=R, u_min=u_min, u_max=u_max,  \
        time_step=control_T, num_nodes=num_nodes, rti=rti,)

    # run a single control loop with trajectory prediction
    x0 = np.zeros(13)
    x0[3] = 1
    # the setpoint must span the entire prediction horizon
    x_set = np.tile(
        np.array([2,-2, 4, 1,0,0,0, 0,0,0, 0,0,0]), num_nodes
    )
    mpc.get_trajectory(x0, x_set, timer=True, visuals=True)


if __name__=="__main__":
    main()

#!/usr/bin/python3

from qrac.dynamics import NonlinearCrazyflie
from qrac.control.acados_mpc import AcadosMpc
import numpy as np


def main():
    # get dynamics model and convert to acados
    model = NonlinearCrazyflie(Ax=0, Ay=0, Az=0)

    # initialize controller
    Q = np.diag([10,10,10, 1,1,1, 2.8,2.8,2.8, 1,1,1,])
    R = 0.1 * np.diag([1, 1, 1, 1])
    max_thrust = 0.64   # N
    u_max = max_thrust * np.ones(4)
    u_min = np.zeros(4)
    control_T = 0.02
    N = 16
    rt = False
    mpc = AcadosMpc(
        model=model, Q=Q, R=R, u_max=u_max, u_min=u_min, \
        time_step=control_T, num_nodes=N, real_time=rt,)

    # run a single control loop with trajectory prediction
    x0 = np.zeros(12)
    x_set = np.array([5,-4,10, 0,0,0, 0,0,0, 0,0,0])
    mpc.get_state(x0, x_set, timer=True, visuals=True)


if __name__=="__main__":
    main()
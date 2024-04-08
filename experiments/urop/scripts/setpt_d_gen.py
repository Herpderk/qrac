#!/usr/bin/python3

import numpy as np
from qrac.models import Crazyflie


def main():
    # sim settings
    PREDICT_TIME = 30
    CTRL_T = 0.01
    NODES = int(round(PREDICT_TIME / CTRL_T))


    # disturbance settings
    D_MAX = np.array([
        0,0,0, 0,0,0, 4,4,4, 4,4,4,
    ])
    dfilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_d.npy"


    # init model
    model = Crazyflie(Ax=0, Ay=0, Az=0)
    nx = model.nx

    # generate uniformly distributed disturbance
    d = D_MAX * np.random.uniform(-1, 1, (NODES, nx))
    np.save(dfilename, d)


if __name__=="__main__":
    main()

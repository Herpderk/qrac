#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def main(SIM_TIME: int):
    CTRL_T = 0.01
    NODES = int(round(SIM_TIME/CTRL_T))

    t = np.arange(0, SIM_TIME, CTRL_T)
    nmpc_err = np.load("/home/derek/dev/my-repos/qrac/experiments/urop/lemniscate/data/nmpc_err.npy")
    l1_err = np.load("/home/derek/dev/my-repos/qrac/experiments/urop/lemniscate/data/l1_err.npy")
    smlms_err = np.load("/home/derek/dev/my-repos/qrac/experiments/urop/lemniscate/data/smlms_err.npy")
    mhe_err = np.load("/home/derek/dev/my-repos/qrac/experiments/urop/lemniscate/data/mhe_err.npy")

    nmpc_err = nmpc_err[:NODES]
    l1_err = l1_err[:NODES]
    smlms_err = smlms_err[:NODES]
    mhe_err = mhe_err[:NODES]

    plt.figure(figsize=(16,9))
    plt.plot(t, nmpc_err, label="NMPC", c="tab:blue")
    plt.plot(t, mhe_err, label="MHE+NMPC", c="tab:green")
    plt.plot(t, l1_err, label="L1+NMPC", c="tab:purple")
    plt.plot(t, smlms_err, label="SMLMS+NMPC", c="tab:red")

    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.title("Lemniscate Tracking Errors")
    plt.legend()

    plt.savefig(f"/home/derek/dev/my-repos/qrac/experiments/urop/lemniscate/figures/lem_err_{SIM_TIME}")
    plt.show()


if __name__=="__main__":
    main(SIM_TIME=30)
    main(SIM_TIME=5)

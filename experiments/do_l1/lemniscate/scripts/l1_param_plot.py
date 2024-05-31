#!/usr/bin/python3

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from consts import A_GAINS

def main(
    st: int,
    et: int,
    T: float,
) -> None:
    start = int(st / T)
    end = int(et / T)

    SAMPLE = 40
    xref = np.tile(np.load("../refs/xref.npy")[start : end, :13], SAMPLE)
    
    RMSE_L1 = []
    RMSE_DO_L1 = []
    ERR_BAR_L1 = np.zeros((2, len(A_GAINS)))
    ERR_BAR_DO_L1 = np.zeros((2, len(A_GAINS)))
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.set_xscale("log")
    
    for j in range(len(A_GAINS)):
        A_GAIN = A_GAINS[j]
        for M in range(2):
            all_traj = np.load(f"../data/traj_M={M}_A={A_GAIN}_I={0}.npy")[start : end]
            for i in range(1, SAMPLE):
                traj = np.load(f"../data/traj_M={M}_A={A_GAIN}_I={i}.npy")[start : end]
                all_traj = np.hstack((all_traj, traj))
            
            err =  np.linalg.norm((xref - all_traj)[:, 0:3], ord=2, axis=1)
            rmse = np.sqrt(np.sum(np.square(err))/err.shape[0])
            if M == 0:
                RMSE_L1 += [rmse]
                ERR_BAR_L1[:,j] = np.percentile(err, [5, 95])
            else:
                RMSE_DO_L1 += [rmse]
                ERR_BAR_DO_L1[:,j] = np.percentile(err, [5, 95])

    ax.errorbar(
        A_GAINS, RMSE_L1, yerr=ERR_BAR_L1,
        fmt="o-", label=f"L1", c="tab:blue",
        markersize=10, capsize=10
    )
    ax.errorbar(
        0.98*np.array(A_GAINS), RMSE_DO_L1, yerr=ERR_BAR_DO_L1,
        fmt="o-", label=f"DO-L1", c="tab:orange",
        markersize=10, capsize=10
    )
    ax.set_xlim(left=10**0.9, right=10**4.6)
    ax.set_ylim(bottom=0, top=2)
    ax.legend()
    
    plt.title("Position RMSE vs. Initial Adaptation Gain")
    plt.xlabel("Initial Adaptation Gain")
    plt.ylabel("Position RMSE w/ 90% Confidence Interval")
    plt.legend()
    plt.show()


if __name__=="__main__":
    T = 0.01
    et = 10
    main(st=0, et=et, T=T)


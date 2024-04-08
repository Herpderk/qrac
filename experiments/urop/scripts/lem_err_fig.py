#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def main(
    st: int,
    et: int,
    T: float,
) -> None:
    figname = f"/home/derek/dev/my-repos/qrac/experiments/urop/figures/lem_err_{st}-{et}.png"

    start = int(round(st / T))
    end = int(round(et / T))
    num = int(round((et-st)/T))

    t = np.linspace(st, et, num)
    xref = np.load("/home/derek/dev/my-repos/qrac/experiments/urop/data/lem_xref.npy")
    nmpc_traj = np.load("/home/derek/dev/my-repos/qrac/experiments/urop/data/lem_nmpc_traj.npy")
    l1_traj = np.load("/home/derek/dev/my-repos/qrac/experiments/urop/data/lem_l1_traj.npy")
    mhe_traj = np.load("/home/derek/dev/my-repos/qrac/experiments/urop/data/lem_mhe_traj.npy")
    smlms_traj = np.load("/home/derek/dev/my-repos/qrac/experiments/urop/data/lem_smlms_traj.npy")
    linmhe_traj = np.load("/home/derek/dev/my-repos/qrac/experiments/urop/data/lem_linmhe_traj.npy")

    nmpc_err = np.linalg.norm((xref - nmpc_traj)[:, 0:3], ord=2, axis=1)[start : end]
    l1_err = np.linalg.norm((xref - l1_traj)[:, 0:3], ord=2, axis=1)[start : end]
    mhe_err = np.linalg.norm((xref - mhe_traj)[:, 0:3], ord=2, axis=1)[start : end]
    smlms_err = np.linalg.norm((xref - smlms_traj)[:, 0:3], ord=2, axis=1)[start : end]
    linmhe_err = np.linalg.norm((xref - linmhe_traj)[:, 0:3], ord=2, axis=1)[start : end]

    nmpc_rmse = np.sqrt(np.sum(np.square(nmpc_err))/nmpc_err.shape[0])
    l1_rmse = np.sqrt(np.sum(np.square(l1_err))/l1_err.shape[0])
    mhe_rmse = np.sqrt(np.sum(np.square(mhe_err))/mhe_err.shape[0])
    smlms_rmse = np.sqrt(np.sum(np.square(smlms_err))/smlms_err.shape[0])
    linmhe_rmse = np.sqrt(np.sum(np.square(linmhe_err))/linmhe_err.shape[0])

    plt.figure(figsize=(16,9))
    plt.plot(t, nmpc_err, label=f"Nominal (RMSE: {round(nmpc_rmse, 3)})", c="black")
    plt.plot(t, l1_err, label=f"L1 (RMSE: {round(l1_rmse, 3)})", c="tab:blue")
    plt.plot(t, mhe_err, label=f"Nonlinear MHE (RMSE: {round(mhe_rmse, 3)}", c="tab:purple")
    plt.plot(t, linmhe_err, label=f"Linear MHE (RMSE: {round(linmhe_rmse, 3)})", c="tab:green")
    plt.plot(t, smlms_err, label=f"SM LMS (RMSE: {round(smlms_rmse, 3)})", c="tab:red")

    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.title("Lemniscate Tracking Errors")
    plt.legend()

    plt.savefig(figname)
    plt.show()


if __name__=="__main__":
    T = 0.01
    et = 30
    step = 10
    main(st=0, et=et, T=T)
    for t in range(0, et, step):
        main(st=t, et=t+step, T=T)

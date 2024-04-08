#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def main(
    st: int,
    et: int,
    T: float,
) -> None:
    figname = f"/home/derek/dev/my-repos/qrac/experiments/urop/figures/setpt_err_zoom_{st}-{et}.png"

    start = int(round(st / T))
    end = int(round(et / T))
    num = int(round((et-st)/T))

    t = np.linspace(st, et, num)
    xref = np.load("/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_xref.npy")
    smlms_traj = np.load("/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_smlms_traj.npy")
    linmhe_traj = np.load("/home/derek/dev/my-repos/qrac/experiments/urop/data/setpt_linmhe_traj.npy")

    smlms_err = np.linalg.norm((xref - smlms_traj)[:, 0:3], ord=2, axis=1)[start : end]
    linmhe_err = np.linalg.norm((xref - linmhe_traj)[:, 0:3], ord=2, axis=1)[start : end]

    smlms_rmse = np.sqrt(np.sum(np.square(smlms_err))/smlms_err.shape[0])
    linmhe_rmse = np.sqrt(np.sum(np.square(linmhe_err))/linmhe_err.shape[0])

    plt.figure(figsize=(16,9))
    plt.plot(t, linmhe_err, label=f"Linear MHE (RMSE: {round(linmhe_rmse, 3)})", c="tab:green")
    plt.plot(t, smlms_err, label=f"SM LMS (RMSE: {round(smlms_rmse, 3)})", c="tab:red")

    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.title("Setpoint Stabilization Errors")
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

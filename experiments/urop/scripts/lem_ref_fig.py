#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def main():
    xfilename = "/home/derek/dev/my-repos/qrac/experiments/urop/data/lem_xref.npy"
    figname = "/home/derek/dev/my-repos/qrac/experiments/urop/figures/lem_xref.png"

    fig = plt.figure(figsize=(9,8))
    ax = fig.add_subplot(projection="3d")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 2)
    ax.xaxis.set_rotate_label(False)

    ax.set_xlabel(r"$\bf{x}$ (m)", fontsize=12)
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel(r"$\bf{y}$ (m)", fontsize=12)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r"$\bf{z}$ (m)", fontsize=12)

    xref = np.load(xfilename)
    x = xref[:,0]
    y = xref[:,1]
    z = xref[:,2]
    line = ax.plot([],[],[], c="r")[0]
    line.set_data_3d(x, y, z)

    plt.savefig(figname)
    plt.show()


if __name__=="__main__":
    main()

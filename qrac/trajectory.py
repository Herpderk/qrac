#!/usr/bin/python3

import casadi as cs
import numpy as np


class Circle():
    def __init__(self, r: float, alt: float):
        t = cs.SX.sym("t")
        traj = cs.SX.sym("traj", 12)
        pos_ang = cs.SX(cs.vertcat(
            r*cs.cos(t),
            r*cs.sin(t),
            alt,
            cs.SX.zeros(3,1)))
        vels = cs.SX(cs.jacobian(pos_ang, t))

        traj[0:6] = pos_ang
        traj[6:12] = vels
        self._traj = cs.Function("trajectory", [t], [traj])

    def get_setpoint(self, t: float):
        setpt = self._traj(t)
        return np.array(setpt).flatten()
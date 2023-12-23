#!/usr/bin/python3

import casadi as cs
import numpy as np


class Circle():
    def __init__(self, v: float, r: float, alt: float):
        t = cs.SX.sym("t")
        k = v / r
        pos_ang = cs.SX(cs.vertcat(
            r*cs.cos(k*t),
            r*cs.sin(k*t),
            alt,
            cs.SX.zeros(3,1)
        ))
        vels = cs.SX(cs.jacobian(pos_ang, t))
        traj = cs.SX(cs.vertcat(
            pos_ang,
            vels,
        ))
        self._traj = cs.Function("trajectory", [t], [traj])

    def get_setpoint(self, t: float):
        setpt = self._traj(t)
        return np.array(setpt).flatten()
#!/usr/bin/python3

import casadi as cs
import numpy as np
from typing import List


class Circle():
    def __init__(
        self,
        v: float,
        r: float,
        alt: float
    ) -> None:
        self._traj = self._init_func(v=v, r=r, alt=alt)
        
    def get_setpoint(
        self,
        t: float
    ) -> np.ndarray:
        setpt = self._traj(t)
        return np.array(setpt).flatten()

    def _init_func(
        self,
        v: float,
        r: float,
        alt: float
    ) -> cs.Function:
        t = cs.SX.sym("t")
        k = v / r
        pos = cs.SX(cs.vertcat(
            r*cs.cos(k*t),
            r*cs.sin(k*t),
            alt,
        ))
        quat = cs.SX(cs.vertcat(
            1, cs.SX.zeros(3)
        ))
        vels = cs.SX(cs.jacobian(pos, t))
        angvels = cs.SX.zeros(3)
        traj = cs.SX(cs.vertcat(
            pos,
            quat,
            vels,
            angvels
        ))
        traj_func = cs.Function("trajectory", [t], [traj])
        return traj_func



class LemniScate():
    def __init__(
        self,
        a: float,
        b: float,
        axes: List,
        translation: List
    ) -> None:
        self._traj = self._init_func(
            a=a, b=b, axes=axes, translation=translation
        )

    def get_setpoint(
        self,
        t: float
    ) -> np.ndarray:
        setpt = self._traj(t)
        return np.array(setpt).flatten()

    def _init_func(
        self,
        a: float,
        b: float,
        axes: List,
        translation: List
    ) -> cs.Function:
        assert len(axes) == 2
        len(translation) == 3
        t = cs.SX.sym("t")
        pos = cs.SX.sym("pos_ang", 3)

        for ax in range(3):
            if ax == axes[0]:
                pos[ax] = translation[ax] + \
                    (a*cs.cos(b*t)) / (1 + (cs.sin(b*t)**2))
            elif ax == axes[1]:
                pos[ax] = translation[ax] + \
                    (a*cs.cos(b*t)*cs.sin(b*t)) / (1 + (cs.sin(b*t)**2))
            else:
                pos[ax] = translation[ax]

        pos_ang = cs.SX(cs.vertcat(
            pos, cs.SX.zeros(3)
        ))
        vels = cs.SX(cs.jacobian(pos_ang, t))
        traj = cs.SX(cs.vertcat(
            pos_ang,
            vels,
        ))
        traj_func = cs.Function("trajectory", [t], [traj])
        return traj_func

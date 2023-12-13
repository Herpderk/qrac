#!/usr/bin/python3

import casadi as cs


def circle(r: float, alt: float):
    t = cs.SX.sym("t")
    pos_ang = cs.SX(cs.vertcat(
        r*cs.cos(t),
        r*cs.sin(t),
        alt,
        cs.SX.zeros(3,1)))
    vels = cs.SX(cs.jacobian(pos_ang, t))
    traj = cs.SX(cs.vertcat(pos_ang, vels))
    traj = cs.reshape(traj, 1, 12)
    return cs.Function("trajectory", [t], [traj])
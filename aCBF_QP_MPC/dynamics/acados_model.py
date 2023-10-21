#!/usr/bin/python3

import casadi as cs
from acados_template import AcadosModel


def get_nonlinear_acados_model(m, l, Ixx, Iyy, Izz, k,):# kf, km, Ax, Ay, Az):
    """
    Returns casadi struct containing explicit dynamics,
    state, state_dot, control input, and name. 
    Nonlinear continuous-time quadrotor dynamics. 
    The cartesian states are in ENU.
    """
    assert (type(m) == int or type(m) == float)
    assert (type(l) == int or type(l) == float)
    assert (type(Ixx) == int or type(Ixx) == float)
    assert (type(Iyy) == int or type(Iyy) == float)
    assert (type(Izz) == int or type(Izz) == float)
    assert (type(k) == int or type(k) == float)

    # State Variables: position, rotation, velocity, and body-frame angular velocity
    x = cs.SX.sym("x")
    y = cs.SX.sym("y")
    z = cs.SX.sym("z")
    phi = cs.SX.sym("phi")     # roll
    theta = cs.SX.sym("theta") # pitch
    psi = cs.SX.sym("psi")     # yaw
    x_d = cs.SX.sym("x_d")     # time-derivatives
    y_d = cs.SX.sym("y_d")
    z_d = cs.SX.sym("z_d")
    phi_d_B = cs.SX.sym("phi_d_B")
    theta_d_B = cs.SX.sym("theta_d_B")
    psi_d_B = cs.SX.sym("psi_d_B")
    X = cs.SX(cs.vertcat(
        x, y, z, phi, theta, psi,\
        x_d, y_d, z_d, phi_d_B, theta_d_B, psi_d_B
    ))


    # rotation matrix from body frame to inertial frame
    Rx = cs.SX(cs.vertcat(
        cs.horzcat(1,            0,            0),
        cs.horzcat(0,  cs.cos(phi), -cs.sin(phi)),
        cs.horzcat(0,  cs.sin(phi),  cs.cos(phi)),
    ))
    Ry = cs.SX(cs.vertcat(
        cs.horzcat( cs.cos(theta),  0,  cs.sin(theta)),
        cs.horzcat(             0,  1,              0),
        cs.horzcat(-cs.sin(theta),  0,  cs.cos(theta)),
    ))
    Rz = cs.SX(cs.vertcat(
        cs.horzcat(cs.cos(psi),    -cs.sin(psi),    0),
        cs.horzcat(cs.sin(psi),     cs.cos(psi),    0),
        cs.horzcat(          0,               0,    1),
    ))
    R = Rz @ Ry @ Rx

    # Diagonal of inertial matrix
    J = cs.SX(cs.vertcat(
        cs.horzcat(Ixx, 0, 0),
        cs.horzcat(0, Iyy, 0),
        cs.horzcat(0, 0, Izz),
    ))

    # thrust of motors 1 to 4
    u = cs.SX.sym("u", 4)

    # continuous-time dynamics
    gravity = 9.81              # acceleration due to gravity
    w_B = cs.SX(cs.vertcat(phi_d_B, theta_d_B, psi_d_B))
    f = cs.SX(cs.vertcat(
        x_d, y_d, z_d,
        phi_d_B, theta_d_B, psi_d_B,
        0,0,-gravity,
        cs.inv(J) @ (cs.cross(-w_B, J @ w_B)),
    ))

    g_bot = cs.SX(cs.vertcat(
        cs.horzcat(R/m, cs.SX.zeros(3,3)),
        cs.horzcat(cs.SX.zeros(3,3), cs.inv(J))
    ))
    u_to_g = cs.SX(cs.vertcat(
        cs.SX.zeros(2,4),
        (1/m) * cs.SX.ones(1,4),
        cs.horzcat(0, -l, 0, l),
        cs.horzcat(-l, 0, l, 0),
        cs.horzcat(-k, k, -k, k),
    ))
    g = cs.SX(cs.vertcat(
        cs.SX.zeros(6, 4),
        g_bot @ u_to_g))

    # control affine formulation
    Xdot = f + g @ u

    # store variables in casadi struct
    model = AcadosModel()
    model.f_expl_expr = Xdot
    model.x = X
    model.xdot = Xdot
    model.u = u 
    model.name = "nonlinear_quadrotor_model"
    return model


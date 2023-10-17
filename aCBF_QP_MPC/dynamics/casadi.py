#!/usr/bin/python3

import casadi as cs
import numpy as np


def nonlinear_quadrotor_model(m, l, Ixx, Iyy, Izz, kf, km, Ax, Ay, Az):
    ''' Returns casadi struct containing explicit dynamics,
        state, state_dot, control input, and name. 
        Nonlinear continuous-time quadrotor dynamics. 
        The cartesian states are in ENU.
    '''
    # State Variables: position, rotation, and their time-derivatives
    x = cs.SX.sym('x')
    y = cs.SX.sym('y')
    z = cs.SX.sym('z')
    phi = cs.SX.sym('phi')     # roll
    theta = cs.SX.sym('theta') # pitch
    psi = cs.SX.sym('psi')     # yaw
    x_d = cs.SX.sym('x_d')     # time-derivatives
    y_d = cs.SX.sym('y_d')
    z_d = cs.SX.sym('z_d')
    phi_d_B = cs.SX.sym('phi_d_B')
    theta_d_B = cs.SX.sym('theta_d_B')
    psi_d_B = cs.SX.sym('psi_d_B')
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


    # calculation of jacobian matrix that converts body frame vels to inertial frame
    I = cs.SX(cs.vertcat(
        cs.horzcat(Ixx, 0, 0),
        cs.horzcat(0, Iyy, 0),
        cs.horzcat(0, 0, Izz),
    ))

    '''
    W = cs.SX(cs.vertcat( 
        cs.horzcat(1,  0,        -cs.sin(theta)),
        cs.horzcat(0,  cs.cos(phi),  cs.cos(theta)*cs.sin(phi)),   
        cs.horzcat(0, -cs.sin(phi),  cs.cos(theta)*cs.cos(phi)),
    ))
    J = W.T @ I @ W


    # Coriolis matrix for defining angular equations of motion
    C11 = 0

    C12 = (Iyy-Izz)*(theta_d*cs.cos(phi)*cs.sin(phi) + psi_d*(cs.sin(phi)**2)*cs.cos(theta)) +\
        (Izz-Iyy)*psi_d*(cs.cos(phi)**2)*cs.cos(theta) -\
        Ixx*psi_d*cs.cos(theta)

    C13 = (Izz-Iyy)*psi_d*cs.cos(phi)*cs.sin(phi)*(cs.cos(theta)**2)

    C21 = (Izz-Iyy)*(theta_d*cs.cos(phi)*cs.sin(phi) + psi_d*(cs.sin(phi)**2)*cs.cos(theta)) +\
        (Iyy-Izz)*psi_d*(cs.cos(phi)**2)*cs.cos(theta) +\
        Ixx*psi_d*cs.cos(theta)

    C22 = (Izz-Iyy)*phi_d*cs.cos(phi)*cs.sin(phi)

    C23 = -Ixx*psi_d*cs.sin(theta)*cs.cos(theta) +\
        Iyy*psi_d*(cs.sin(phi)**2)*cs.sin(theta)*cs.cos(theta) +\
        Izz*psi_d*(cs.cos(phi)**2)*cs.sin(theta)*cs.cos(theta)

    C31 = (Iyy-Izz)*psi_d*(cs.cos(theta)**2)*cs.sin(phi)*cs.cos(phi) -\
        Ixx*theta_d*cs.cos(theta)

    C32 = (Izz-Iyy)*(theta_d*cs.cos(phi)*cs.sin(phi)*cs.sin(theta) + phi_d*(cs.sin(phi)**2)*cs.cos(theta)) +\
        (Iyy-Izz)*phi_d*(cs.cos(phi)**2)*cs.cos(theta) +\
        Ixx*psi_d*cs.sin(theta)*cs.cos(theta) -\
        Iyy*psi_d*(cs.sin(phi)**2)*cs.sin(theta)*cs.cos(theta) -\
        Izz*psi_d*(cs.cos(phi)**2)*cs.sin(theta)*cs.cos(theta)

    C33 = (Iyy-Izz)*phi_d*cs.cos(phi)*cs.sin(phi)*(cs.cos(theta)**2) -\
        Iyy*theta_d*(cs.sin(phi)**2)*cs.cos(theta)*cs.sin(theta) -\
        Izz*theta_d*(cs.cos(phi)**2)*cs.cos(theta)*cs.sin(theta) +\
        Ixx*theta_d*cs.cos(theta)*cs.sin(theta)

    C = cs.SX(cs.vertcat(
        cs.horzcat(C11, C12, C13), 
        cs.horzcat(C21, C22, C23), 
        cs.horzcat(C31, C32, C33),
    ))
    '''

    # thrust, tau_x, tau_y, tau_z
    u = cs.SX.sym('u', 4)


    # continuous-time dynamics
    gravity = 9.81              # acceleration due to gravity
    w_B = cs.SX(cs.vertcat(phi_d_B, theta_d_B, psi_d_B))
    f = cs.SX(cs.vertcat(
        x_d, y_d, z_d,
        phi_d_B, theta_d_B, psi_d_B,
        0,0,-gravity,
        cs.inv(I) @ (cs.cross(-w_B, I @ w_B)),
        #cs.inv(J) @ C @ -cs.vertcat(phi_d, theta_d, psi_d)
    ))
    '''
    w_to_T = cs.SX(cs.vertcat(
        cs.SX.zeros(2,4),
        cs.horzcat(kf/m, kf/m, kf/m, kf/m),
        cs.horzcat(0, -l*kf, 0, l*kf),
        cs.horzcat(-l*kf, 0, l*kf, 0),
        cs.horzcat(-km, km, -km, km),
    ))
    '''
    g_bot = cs.SX(cs.vertcat(
        cs.horzcat(R/m, cs.SX.zeros(3,3)),
        cs.horzcat(cs.SX.zeros(3,3), cs.inv(I))
    ))
    u_to_g = cs.SX(cs.vertcat(
        cs.SX.zeros(2,4),
        cs.SX.eye(4)
    ))

    g = cs.SX(cs.vertcat(
        cs.SX.zeros(6, 4),
        g_bot @ u_to_g
    ))

    Xdot = f + g @ u

    # store variables in casadi struct
    model_cs = cs.types.SimpleNamespace()
    model_cs.f_expl_expr = Xdot
    model_cs.x = X
    model_cs.xdot = Xdot
    model_cs.u = u
    model_cs.name = 'nonlin_quadrotor'
    return model_cs


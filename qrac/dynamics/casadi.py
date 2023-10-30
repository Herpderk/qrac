#!/usr/bin/python3

import casadi as cs


class CasadiModel():
    def __init__(self, f_expl_expr, x, xdot, u, name,) -> None:
        self.f_expl_expr = f_expl_expr
        self.x = x
        self.xdot = xdot
        self.u = u
        self.name = name


def NonlinearQuadrotor(
    m: float,
    l1: float,
    l2: float,
    l3: float,
    l4: float,
    Ixx: float,
    Iyy: float,
    Izz: float,
    k: float,
    Ax: float,
    Ay: float,
    Az: float,
) -> CasadiModel:
    """
    Returns casadi struct containing explicit dynamics,
    state, state_dot, control input, and name. 
    Nonlinear continuous-time quadrotor dynamics. 
    The cartesian states are in ENU.
    """
    for arg in [m, l1, l2, l3, l4, Ixx, Iyy, Izz, k, Ax, Ay, Az]:
        assert (type(arg) == int or type(arg) == float)

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
    gravity = cs.SX(cs.vertcat(0, 0, -9.81))
    v = cs.SX(cs.vertcat(x_d, y_d, z_d))
    w_B = cs.SX(cs.vertcat(phi_d_B, theta_d_B, psi_d_B))
    A = cs.SX(cs.vertcat(
        cs.horzcat(Ax, 0, 0),
        cs.horzcat(0, Ay, 0),
        cs.horzcat(0, 0, Az),
    ))
    f = cs.SX(cs.vertcat(
        v,
        w_B,
        gravity - A @ v,
        cs.inv(J) @ (cs.cross(-w_B, J @ w_B)),
    ))

    g_bot = cs.SX(cs.vertcat(
        cs.horzcat(R/m, cs.SX.zeros(3,3)),
        cs.horzcat(cs.SX.zeros(3,3), cs.inv(J))
    ))
    u_to_g = cs.SX(cs.vertcat(
        cs.SX.zeros(2,4),
        (1/m) * cs.SX.ones(1,4),
        cs.horzcat(0, -l2, 0, l4),
        cs.horzcat(-l1, 0, l3, 0),
        cs.horzcat(-k, k, -k, k),
    ))
    g = cs.SX(cs.vertcat(
        cs.SX.zeros(6, 4),
        g_bot @ u_to_g))

    # control affine formulation
    Xdot = f + g @ u
    model = CasadiModel(
        f_expl_expr=Xdot, x=X, xdot=Xdot, u=u, name="nonlinear_quadrotor")
    return model


def NonlinearCrazyflie(
    Ax: float,
    Ay: float,
    Az: float,
) -> CasadiModel:
    """
    crazyflie system identification:
    https://www.research-collection.ethz.ch/handle/20.500.11850/214143
    """
    m = 0.028                   # kg
    l1 = 0.040                   # m
    l2 = 0.040
    l3 = 0.040
    l4 = 0.040
    Ixx = 3.144988 * 10**(-5)
    Iyy = 3.151127 * 10**(-5)
    Izz = 7.058874 * 10**(-5)
    k = 0.005964552             # k is the ratio of torque to thrust
    Ax = 0
    Ay = 0
    Az = 0
    return NonlinearQuadrotor(
        m, l1, l2, l3, l4, Ixx, Iyy, Izz, k, Ax, Ay, Az)
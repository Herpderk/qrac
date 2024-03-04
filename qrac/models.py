#!/usr/bin/python3

import casadi as cs
import numpy as np
from typing import Tuple
from acados_template import AcadosModel


class Quadrotor():
    def __init__(
        self,
        m: float,
        Ixx: float,
        Iyy: float,
        Izz: float,
        Ax: float,
        Ay: float,
        Az: float,
        xB: np.ndarray,
        yB: np.ndarray,
        k: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
        name="Nonlinear_Quadrotor",
    ) -> None:
        """
        Struct containing mass, moments of inertias, linear air resistance terms,
        distances of rotors from the body-frame x-axis, distances of rotors from the body-frame y-axis,
        and torque coefficient magnitudes associated with each rotor's thrust.
        """
        self._assert(m, Ixx, Iyy, Izz, Ax, Ay, Az, xB, yB, k, name)
        self.m = m
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az
        self.xB = xB
        self.yB = yB
        self.k = k
        self.u_min = u_min
        self.u_max = u_max
        self.name = name
        self._get_dynamics()
        self.nx, self.nu = self.get_dims()


    def get_acados_model(self) -> AcadosModel:
        model_ac = AcadosModel()
        model_ac.f_expl_expr = self.xdot
        model_ac.x = self.x
        model_ac.xdot = self.xdot
        model_ac.u = self.u
        model_ac.name = self.name
        return model_ac


    def get_dims(self) -> Tuple[float]:
        nx = self.x.shape[0]
        nu = self.u.shape[0]
        return nx, nu


    def _get_dynamics(self) -> None:
        # State Variables: position, rotation, velocity, and body-frame angular velocity
        x = cs.SX.sym("x")
        y = cs.SX.sym("y")
        z = cs.SX.sym("z")
        phi = cs.SX.sym("phi")     # roll
        theta = cs.SX.sym("theta") # pitch
        psi = cs.SX.sym("psi")     # yaw
        x_dot = cs.SX.sym("x_dot")     # time-derivatives
        y_dot = cs.SX.sym("y_dot")
        z_dot = cs.SX.sym("z_dot")
        p = cs.SX.sym("p")
        q = cs.SX.sym("q")
        r = cs.SX.sym("r")
        self.x = cs.SX(cs.vertcat(
            x, y, z, phi, theta, psi,\
            x_dot, y_dot, z_dot, p, q, r
        ))

        # transformation from inertial to body frame ang vel
        self.W = cs.SX(cs.vertcat(
            cs.horzcat(1, 0, -cs.sin(theta)),
            cs.horzcat(0, cs.cos(phi), cs.cos(theta)*cs.sin(phi)),
            cs.horzcat(0, -cs.sin(phi), cs.cos(theta)*cs.cos(phi)),
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
        self.R = Rz @ Ry @ Rx

        # drag terms
        self.A = cs.SX(np.diag([self.Ax, self.Ay, self.Az]))

        # Diagonal of inertial matrix
        self.J = cs.SX(np.diag([self.Ixx, self.Iyy, self.Izz]))

        # control allocation matrix
        self.B = cs.SX(cs.vertcat(
            self.yB.reshape(1, self.yB.shape[0]),
            -self.xB.reshape(1, self.xB.shape[0]),
            cs.horzcat(-self.k[0], self.k[1], -self.k[2], self.k[3]),
        ))

        # gravity vector
        self.g = cs.SX(cs.vertcat(0, 0, -9.81))

        # thrust of motors 1 to 4
        self.u = cs.SX.sym("u", 4)
        self.T = cs.SX(cs.vertcat(
            0, 0, self.u[0]+self.u[1]+self.u[2]+self.u[3]))

        # continuous-time dynamics
        v = cs.SX(cs.vertcat(x_dot, y_dot, z_dot))
        w_B = cs.SX(cs.vertcat(p, q, r))
        self.xdot = cs.SX(cs.vertcat(
            v,
            cs.inv(self.W) @ w_B,
            (self.R@self.T - self.A@v)/self.m + self.g,
            cs.inv(self.J) @ (self.B@self.u - cs.cross(w_B, self.J@w_B))
        ))


    def _assert(
        self,
        m: float,
        Ixx: float,
        Iyy: float,
        Izz: float,
        Ax: float,
        Ay: float,
        Az: float,
        xB: np.ndarray,
        yB: np.ndarray,
        k: np.ndarray,
        name: str,
    ) -> None:
        for arg in [m, Ixx, Iyy, Izz, Ax, Ay, Az]:
            if (type(arg) != int and type(arg) != float):
                raise TypeError(f"{arg} should be an int or float!")
        for arg in [xB, yB, k]:
            if len(arg) != 4:
                raise ValueError(f"{arg} should be a float vector of length 4!")
        if type(name) != str:
            raise TypeError("The name should be a string!")


class ParameterAffineQuadrotor(Quadrotor):
    def __init__(
        self,
        model: Quadrotor
    ) -> None:
        super().__init__(
            model.m, model.Ixx, model.Iyy, model.Izz,
            model.Ax, model.Ay, model.Az, model.xB, model.yB,
            model.k, model.u_min, model.u_max, "Parameter_Affine_Quadrotor"
        )
        assert type(model) == Quadrotor
        self.nparam = 10
        self.param_idx = model.nx
        self._get_param_affine_dynamics()


    def get_parameters(self) -> np.ndarray:
        param = np.array([
            1/self.m, self.Ax/self.m,
            self.Ay/self.m, self.Az/self.m,
            1/self.Ixx, 1/self.Iyy, 1/self.Izz,
            (self.Izz-self.Iyy)/self.Ixx,
            (self.Ixx-self.Izz)/self.Iyy,
            (self.Iyy-self.Ixx)/self.Izz
        ])
        return param


    def _get_param_affine_dynamics(self) -> None:
        param = cs.SX.sym("param", self.nparam)
        x_aug = cs.SX(cs.vertcat(
            self.x, param
        ))

        vels = self.x[6:9]
        p = x_aug[9]
        q = x_aug[10]
        r = x_aug[11]

        self.F = cs.SX(cs.vertcat(
            vels,
            cs.inv(self.W) @ cs.vertcat(p,q,r),
            self.g,
            cs.SX.zeros(3),
        ))
        self.G = cs.SX(cs.vertcat(
            cs.SX.zeros(6, 10),
            cs.horzcat(
                self.R @ self.T,
                cs.diag(-vels),
                cs.SX.zeros(3, 6)
            ),
            cs.horzcat(
                cs.SX.zeros(3, 4), 
                cs.diag(self.B @ self.u),
                -cs.diag(cs.vertcat(q*r, p*r, p*q))
            ),
        ))

        F_aug = cs.SX(cs.vertcat(
            self.F, cs.SX.zeros(10)
        ))
        G_aug = cs.SX(cs.vertcat(
            self.G, cs.SX.zeros(10,10)
        ))

        self.x = x_aug
        self.xdot = F_aug + G_aug @ param
        self.nx, self.nu = self.get_dims()


class DisturbedQuadrotor(Quadrotor):
    def __init__(
            self,
            model: Quadrotor
        ) -> None:
            super().__init__(
                model.m, model.Ixx, model.Iyy, model.Izz,
                model.Ax, model.Ay, model.Az, model.xB, model.yB,
                model.k, model.u_min, model.u_max, "Disturbed_Quadrotor"
            )
            assert type(model) == Quadrotor
            self.nd = 12
            self._get_disturbed_dynamics()


    def _get_disturbed_dynamics(self) -> None:
        d = cs.SX.sym("disturbance", self.nd)
        x_aug = cs.SX(cs.vertcat(self.x, d))
        xdot_aug = cs.SX(cs.vertcat(
            self.xdot[:self.nx] + d,
            cs.SX.zeros(self.nd)
        ))
        self.x = x_aug
        self.xdot = xdot_aug


def Crazyflie(
    Ax: float,
    Ay: float,
    Az: float,
) -> Quadrotor:
    """
    crazyflie system identification:
    https://www.research-collection.ethz.ch/handle/20.500.11850/214143
    """
    m = 0.028
    Ixx = 3.144988 * 10**(-5)
    Iyy = 3.151127 * 10**(-5)
    Izz = 7.058874 * 10**(-5)
    xB = np.array(
        [0.0283, 0.0283, -0.0283, -0.0283])
    yB = np.array(
        [0.0283, -0.0283, -0.0283, 0.0283])
    k = 0.005964552 * np.ones(4)
    u_min = -0.15 * np.ones(4)
    u_max = -u_min
    return Quadrotor(
        m, Ixx, Iyy, Izz, Ax, Ay, Az, xB, yB, k, u_min, u_max)

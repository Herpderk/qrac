#!/usr/bin/python3

from typing import Tuple
import casadi as cs
import numpy as np
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
        kf: float,
        km: float,
        xB: np.ndarray,
        yB: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
        name="Nonlinear_Quadrotor",
    ) -> None:
        """
        Struct containing mass, moments of inertias, linear air resistance terms,
        thrust and moment coefficients, distances of rotors from the body-frame x-axis,
        and distances of rotors from the body-frame y-axis,
        """
        self._assert(
            m=m, Ixx=Ixx, Iyy=Iyy, Izz=Izz,
            Ax=Ax, Ay=Ay, Az=Az, kf=kf, km=km,
            xB=xB, yB=yB, name=name
        )
        self.m = m
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az
        self.xB = xB
        self.yB = yB
        self.kf = kf
        self.km = km
        self.u_min = u_min
        self.u_max = u_max
        self.name = name
        self._get_dynamics()
        self.np = 0
        self.nx, self.nu = self.get_dims()

    def get_acados_model(self) -> AcadosModel:
        model_ac = AcadosModel()
        model_ac.f_expl_expr = self.xdot
        model_ac.x = self.x
        model_ac.xdot = self.xdot
        model_ac.u = self.u
        model_ac.p = self.p
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
            self.kf * self.yB.reshape(1, self.yB.shape[0]),
            self.kf * -self.xB.reshape(1, self.xB.shape[0]),
            self.km * cs.horzcat(-1, 1, -1, 1),
        ))

        # gravity vector
        self.g = cs.SX(cs.vertcat(0, 0, -9.81))

        # thrust of motors 1 to 4
        self.u = cs.SX.sym("u", 4)
        self.T = cs.SX(cs.vertcat(
            0, 0, self.kf * (self.u[0]+self.u[1]+self.u[2]+self.u[3])
        ))

        # continuous-time dynamics
        v = cs.SX(cs.vertcat(x_dot, y_dot, z_dot))
        w_B = cs.SX(cs.vertcat(p, q, r))
        self.xdot = cs.SX(cs.vertcat(
            v,
            cs.inv(self.W) @ w_B,
            (self.R@self.T - self.A@v)/self.m + self.g,
            cs.inv(self.J) @ (self.B@self.u - cs.cross(w_B, self.J@w_B))
        ))
        
        # ocp problem parameter
        self.p = np.array([])

    def _assert(
        self,
        m: float,
        Ixx: float,
        Iyy: float,
        Izz: float,
        Ax: float,
        Ay: float,
        Az: float,
        kf: float,
        km: float,
        xB: np.ndarray,
        yB: np.ndarray,
        name: str,
    ) -> None:
        for arg in [m, Ixx, Iyy, Izz, Ax, Ay, Az, kf, km]:
            if (type(arg) != int and type(arg) != float):
                raise TypeError(f"{arg} should be an int or float!")
        for arg in [xB, yB,]:
            if len(arg) != 4:
                raise ValueError(f"{arg} should be a float vector of length 4!")
        if type(name) != str:
            raise TypeError("The name should be a string!")


class ParameterizedQuadrotor(Quadrotor):
    def __init__(
        self,
        model: Quadrotor
    ) -> None:
        assert type(model) == Quadrotor
        super().__init__(
            model.m, model.Ixx, model.Iyy, model.Izz,
            model.Ax, model.Ay, model.Az, model.kf, model.km,
            model.xB, model.yB, model.u_min, model.u_max,
            "Nonlinear_Parameterized_Quadrotor"
        )
        self.np = 4
        self._get_param_dynamics()

    def get_parameters(self) -> np.ndarray:
        param = np.array([
            self.m, #self.Ax, self.Ay, self.Az,
            self.Ixx, self.Iyy, self.Izz
        ])
        return param

    def set_prediction_model(self) -> None:
        self.x = cs.SX(cs.vertcat(
            self.x, self.p
        ))
        self.xdot = cs.SX(cs.vertcat(
            self.xdot, cs.SX.zeros(self.np)
        ))
        self.nx, self.nu = self.get_dims()

        self.nu = self.nx - self.np
        self.p = self.u
        d = cs.SX.sym("d", self.nu)
        self.u = d
        self.xdot[:self.nu] += d

    def _get_param_dynamics(self) -> None:
        self.p = cs.SX.sym("p", self.np)
        m = self.p[0]
        A = cs.diag(cs.vertcat(self.Ax, self.Ay, self.Az))
        J = cs.SX(cs.diag(self.p[1:4]))

        # continuous-time dynamics
        v = self.x[6:9]
        w_B = self.x[9:12]
        self.xdot[6:12] = cs.SX(cs.vertcat(
            (self.R@self.T - A@v)/m + self.g,
            cs.inv(J) @ (self.B@self.u - cs.cross(w_B, J@w_B))
        ))


class AffineQuadrotor(Quadrotor):
    def __init__(
        self,
        model: Quadrotor
    ) -> None:
        assert type(model) == Quadrotor
        super().__init__(
            model.m, model.Ixx, model.Iyy, model.Izz,
            model.Ax, model.Ay, model.Az, model.kf, model.km,
            model.xB, model.yB, model.u_min, model.u_max,
            "Parameter_Affine_Quadrotor"
        )
        #self.np = 10
        self.np = 7
        self._get_param_affine_dynamics()

    def get_parameters(self) -> np.ndarray:
        param = np.array([
            1/self.m, #self.Ax/self.m,
            #self.Ay/self.m, self.Az/self.m,
            1/self.Ixx, 1/self.Iyy, 1/self.Izz,
            (self.Izz-self.Iyy)/self.Ixx,
            (self.Ixx-self.Izz)/self.Iyy,
            (self.Iyy-self.Ixx)/self.Izz
        ])
        return param
    
    def set_prediction_model(self) -> None:
        self.x = cs.SX(cs.vertcat(
            self.x, self.p
        ))
        self.xdot = cs.SX(cs.vertcat(
            self.xdot, cs.SX.zeros(self.np)
        ))
        self.nx, self.nu = self.get_dims()

        self.nu = self.nx - self.np
        self.p = self.u
        d = cs.SX.sym("d", self.nu)
        self.u = d
        self.xdot[:self.nu] += d

    def _get_param_affine_dynamics(self) -> None:
        self.p = cs.SX.sym("p", self.np)

        vels = self.x[6:9]
        p = self.x[9]
        q = self.x[10]
        r = self.x[11]
        A = cs.diag(cs.vertcat(self.Ax, self.Ay, self.Az))

        self.F = cs.SX(cs.vertcat(
            vels,
            cs.inv(self.W) @ cs.vertcat(p,q,r),
            self.g,
            cs.SX.zeros(3),
        ))
        '''
        self.G = cs.SX(cs.vertcat(
            cs.SX.zeros(6, self.np),
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
        ))'''
        self.G = cs.SX(cs.vertcat(
            cs.SX.zeros(6, self.np),
            cs.horzcat(
                self.R@self.T - A@vels,
                cs.SX.zeros(3, self.np-1)
            ),
            cs.horzcat(
                cs.SX.zeros(3, self.np-6),
                cs.diag(self.B @ self.u),
                -cs.diag(cs.vertcat(q*r, p*r, p*q))
            ),
        ))
        self.xdot = self.F + self.G @ self.p


class DisturbedQuadrotor(Quadrotor):
    def __init__(
            self,
            model: Quadrotor
        ) -> None:
            assert type(model) == Quadrotor
            super().__init__(
                model.m, model.Ixx, model.Iyy, model.Izz,
                model.Ax, model.Ay, model.Az, model.kf, model.km,
                model.xB, model.yB, model.u_min, model.u_max,
                "Disturbed_Quadrotor"
            )
            self._get_disturbed_dynamics()

    def _get_disturbed_dynamics(self) -> None:
        d = cs.SX.sym("disturbance", self.nx)
        x_aug = cs.SX(cs.vertcat(self.x, d))
        xdot_aug = cs.SX(cs.vertcat(
            self.xdot[:self.nx] + d,
            cs.SX.zeros(self.nx)
        ))
        self.x = x_aug
        self.xdot = xdot_aug


def Crazyflie(
    Ax: float,
    Ay: float,
    Az: float,
    THRUST_MODE=True
) -> Quadrotor:
    """
    crazyflie system identification:
    https://www.research-collection.ethz.ch/handle/20.500.11850/214143
    """
    m = 0.027
    Ixx = 1.6571710 * 10**-5
    Iyy = 1.6655602 * 10**-5
    Izz = 2.9261652 * 10**-5
    xB = 0.0283 * np.array(
        [1, 1, -1, -1])
    yB = 0.0283 * np.array(
        [1, -1, -1, 1])

    if THRUST_MODE:
        kf = 1
        u_max = 0.15 * np.ones(4)
    else:
        # performed additional quadratic fit with root at origin (ax^2) on ethz sys id
        kf = 1.8 * 10**-8
        u_max = 2900 * np.ones(4)

    km = 0.005964552 * kf
    u_min = np.zeros(4)
    return Quadrotor(
        m=m, Ixx=Ixx, Iyy=Iyy, Izz=Izz,
        Ax=Ax, Ay=Ay, Az=Az, kf=kf, km=km,
        xB=xB, yB=yB, u_min=u_min, u_max=u_max
    )

#!/usr/bin/python3

from typing import Tuple
import casadi as cs
import numpy as np
from scipy.linalg import expm
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
        #phi = cs.SX.sym("phi")     # roll
        #theta = cs.SX.sym("theta") # pitch
        #psi = cs.SX.sym("psi")     # yaw
        q0 = cs.SX.sym("q0")
        q1 = cs.SX.sym("q1")
        q2 = cs.SX.sym("q2")
        q3 = cs.SX.sym("q3")
        x_dot = cs.SX.sym("x_dot")     # time-derivatives
        y_dot = cs.SX.sym("y_dot")
        z_dot = cs.SX.sym("z_dot")
        p = cs.SX.sym("p")
        q = cs.SX.sym("q")
        r = cs.SX.sym("r")
        self.x = cs.SX(cs.vertcat(
            x, y, z, q0, q1, q2, q3,
            x_dot, y_dot, z_dot, p, q, r
        ))

        self.R = cs.SX(cs.vertcat(
            cs.horzcat( 1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2) ),
            cs.horzcat( 2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1) ),
            cs.horzcat( 2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2) ),
        ))
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
        qv = cs.SX(cs.vertcat(q1,q2,q3))
        v = cs.SX(cs.vertcat(x_dot, y_dot, z_dot))
        wB = cs.SX(cs.vertcat(p, q, r))
        self.xdot = cs.SX(cs.vertcat(
            v,
            #cs.inv(self.W) @ wB,
            -cs.dot(qv, 0.5*wB),
            q0*0.5*wB + cs.cross(qv,wB),
            (self.R@self.T - self.A@v)/self.m + self.g,
            cs.inv(self.J) @ (self.B@self.u - cs.cross(wB, self.J@wB))
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
        self.np = 7
        self._get_param_dynamics()

    def get_parameters(self) -> np.ndarray:
        param = np.array([
            self.m, self.Ax, self.Ay, self.Az,
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
        A = cs.diag(self.p[1:4])
        J = cs.diag(self.p[4:7])

        # continuous-time dynamics
        v = self.x[self.nx-6 : self.nx-3]
        wB = self.x[self.nx-3 : self.nx]
        self.xdot[self.nx-6 : self.nx] = cs.SX(cs.vertcat(
            (self.R@self.T - A@v)/m + self.g,
            cs.inv(J) @ (self.B@self.u - cs.cross(wB, J@wB))
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
        self.np = 10
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
        q0 = self.x[self.nx-10]
        qv = self.x[self.nx-9 : self.nx-6]
        vels = self.x[self.nx-6 : self.nx-3]
        wB = self.x[self.nx-3 : self.nx]
        p = wB[0]
        q = wB[1]
        r = wB[2]
        
        self.F = cs.SX(cs.vertcat(
            vels,
            -cs.dot(qv, 0.5*wB),
            q0*0.5*wB + cs.cross(qv,wB),
            self.g,
            cs.SX.zeros(3),
        ))
        self.G = cs.SX(cs.vertcat(
            cs.SX.zeros(self.nx-6, self.np),
            cs.horzcat(
                self.R@self.T,
                cs.diag(-vels),
                cs.SX.zeros(3, self.np-4)
            ),
            cs.horzcat(
                cs.SX.zeros(3, self.np-6),
                cs.diag(self.B @ self.u),
                -cs.diag(cs.vertcat(q*r, p*r, p*q))
            ),
        ))
        
        self.p = cs.SX.sym("p", self.np)
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


class L1Quadrotor(Quadrotor):
    def __init__(
        self,
        model: Quadrotor
    ) -> None:
        assert type(model) == Quadrotor
        super().__init__(
            model.m, model.Ixx, model.Iyy, model.Izz,
            model.Ax, model.Ay, model.Az, model.kf, model.km,
            model.xB, model.yB, model.u_min, model.u_max,
            "L1_Quadrotor"
        )
        self.nz = 6
        self.n_um = 2

    def get_l1_opt_dynamics(
        self,
        M: int,
        Ts: float,
        Am: np.ndarray,
    ) -> AcadosModel:
        a_gain = cs.SX.sym("a_gain")
        w = cs.SX.sym("w")
        u = cs.SX(cs.vertcat(
            a_gain, w
        ))
        x = cs.vertcat()
        xdot = cs.vertcat()
        p = cs.vertcat()
        
        for k in range(M):
            zpred = cs.SX.sym(f"zpred_{k}", self.nz)
            ztrue = cs.SX.sym(f"ztrue_{k}", self.nz)
            ul1prev = cs.SX.sym(f"ul1prev_{k}", self.nu)
            unom = cs.SX.sym(f"unom_{k}", self.nu)
            
            f, g_m, g_um = self.get_l1_dynamics(z=ztrue, u=unom)
            G = cs.horzcat(g_m, g_um)
            phi = np.linalg.inv(Am) @ (expm(Am*Ts) - np.eye(self.nz))
            mu = expm(Am*Ts) * (zpred - ztrue)
            
            d = -a_gain @ np.eye(self.nz) @ cs.inv(G) @ cs.inv(phi) @ mu
            d_m = d[:self.nu]
            d_um = d[self.nu : self.nu + self.n_um]

            ul1 = ul1prev*np.exp(-w*Ts) - d_m*(1-np.exp(-w*Ts))
            #utot = unom + ul1

            zpred_dot = Am@(zpred - ztrue) + f + g_m@(ul1+d_m) + g_um@d_um
            #utot_dot = cs.SX.zeros(3)
            
            x = cs.SX(cs.vertcat(
                x, zpred, #utot
            ))
            xdot = cs.SX(cs.vertcat(
                xdot, zpred_dot, #utot_dot
            ))
            p = cs.SX(cs.vertcat(
                p, ztrue, unom, ul1prev
            ))

        self.x = x
        self.xdot = xdot
        self.u = u
        self.p = p
        self.np = p.shape[0]
        self.nx, self.nu = self.get_dims()


    def get_l1_dynamics(
        self,
        z: cs.SX,
        u: cs.SX,
    ) -> Tuple[cs.SX, cs.SX, cs.SX]:
        assert len(z) == self.nz
        assert len(u) == self.nu
        b1 = self.R[:,0]
        b2 = self.R[:,1]
        b3 = self.R[:,2]

        f = cs.SX(cs.vertcat(
            self.R@cs.vertcat(0,0,cs.sum1(u)) - self.A@z[0:3])/self.m + self.g,
            cs.inv(self.J) @ (self.B@u - cs.cross(z[3:6], self.J@z[3:6])
        ))
        g_m = cs.SX(cs.vertcat(
            b3/self.m @ cs.SX.ones(1,self.nu),
            cs.inv(self.J) @ self.B
        ))
        g_um = cs.SX(cs.vertcat(
            cs.horzcat(b1, b2)/self.m,
            cs.SX.zeros(self.nz-3,self.n_um)
        ))
        return f, g_m, g_um


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

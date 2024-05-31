#!/usr/bin/python3

from typing import Tuple
import casadi as cs
import numpy as np
from scipy.linalg import expm, block_diag
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

    def get_implicit_model(self) -> AcadosModel:
        model_ac = AcadosModel()
        model_ac.f_expl_expr = self.f_expl_expr
        model_ac.f_impl_expr = self.f_impl_expr
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
            if (type(arg) != int and type(arg) != float and type(arg) != np.float64):
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
        self.nx, self.nu = self.get_dims()


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
        self.nx, self.nu = self.get_dims()


# move args (M, Ts, Am) to a manually called function get_discrete_model
class L1Quadrotor(Quadrotor):
    def __init__(
        self,
        model: Quadrotor,
    ) -> None:
        assert type(model) == Quadrotor
        super().__init__(
            model.m, model.Ixx, model.Iyy, model.Izz,
            model.Ax, model.Ay, model.Az, model.kf, model.km,
            model.xB, model.yB, model.u_min, model.u_max,
            "L1_Quadrotor"
        )
        self.nz = 6
        self.ny = self.nz + self.nu
        self.n_m = 4
        self.n_um = 2

    def get_predictor_funcs(self) -> Tuple[cs.Function, cs.Function, cs.Function]:
        upred = cs.SX.sym("upred", 4)
        f, g_m, g_um = self._get_predictor_vars(x=self.x, u=upred)
        f_func = cs.Function("f", [self.x, upred], [f])
        g_m_func = cs.Function("g_m", [self.x], [g_m])
        g_um_func = cs.Function("g_um", [self.x], [g_um])
        return f_func, g_m_func, g_um_func

    def get_discrete_model(
        self,
        Ts: float,
        Am: np.ndarray,
    ) -> AcadosModel:
        self._get_l1_opt_dynamics(Ts=Ts, Am=Am)
        model_ac = AcadosModel()
        model_ac.disc_dyn_expr = self.disc_dyn_expr
        model_ac.x = self.x
        model_ac.u = self.u
        model_ac.p = self.p
        model_ac.name = self.name
        return model_ac

    def get_casadi_solver(
        self,
        Ts: float,
        Am: np.ndarray,
        Q: float,
    ):
        wb = cs.SX.sym("wb")
        a_gain = cs.SX.sym("a_gain")
        a_gain_prev = cs.SX.sym("a_gain_prev")
        z_nxt = cs.SX.sym("z_nxt", self.nz)
        p, zcost = self._get_l1_opt_dynamics(
            Ts=Ts, a_gain=a_gain, wb=wb, Am=Am
        )
        
        p = cs.vertcat(p, a_gain_prev, z_nxt,)
        self.np = p.shape[0]

        R = cs.horzcat( 
            cs.vertcat(self.b3, cs.SX.zeros(1)),
            cs.vertcat(cs.SX.zeros(1,3), cs.SX.eye(3)),
        )
        f = (cs.inv(R) @ (zcost - z_nxt)[-self.n_m:]).T @ \
            (cs.inv(R) @ (zcost - z_nxt)[-self.n_m:]) \
            + Q * (a_gain - a_gain_prev)**2

        qp = {"x": a_gain, "p": p, "f": f}
        opts = {"verbose": False}
        sol = cs.nlpsol("sol", "ipopt", qp, opts)
        return sol

    def _get_l1_opt_dynamics(
        self,
        Ts: float,
        Am: cs.SX,
        a_gain: cs.SX,
        wb: cs.SX,
    ) -> Tuple[cs.SX, cs.SX, cs.SX]:
        zpred = cs.SX.sym("zpred_opt_l1", self.nz)
        ul1prev = cs.SX.sym("ul1prev_opt_l1", self.nu)
        ul1curr = cs.SX.sym("ul1curr_opt_l1", self.nu)
        utot = cs.SX.sym("utot_opt_l1", self.nu)
        xtrue = cs.SX.sym("x_opt_l1", self.nx)
        ztrue = xtrue[-self.nz:]
        unom = utot - ul1curr

        f, g_m, g_um = self._get_predictor_vars(x=xtrue, u=unom)
        G = cs.horzcat(g_m, g_um)
        phi = cs.inv(Am) @ cs.SX(expm(Am*Ts) - np.eye(self.nz))
        mu = expm(Am*Ts) * (zpred - ztrue)

        d = -a_gain * cs.SX.eye(self.nz) @ cs.inv(G) @ cs.inv(phi) @ mu
        d_m = d[:self.n_m]
        d_um = d[-self.n_um:]

        ul1_nxt = ul1prev*cs.exp(-wb*Ts) - d_m*(1-cs.exp(-wb*Ts))

        # predictor in cost
        zcost = ztrue + Ts * (
            f + g_m @ (ul1_nxt + d_m)
        )
        p = cs.SX(cs.vertcat(utot, ul1prev, ul1curr, xtrue, zpred, wb,))
        return p, zcost

    def _get_predictor_vars(
        self,
        x: cs.SX,
        u: cs.SX,
    ) -> Tuple[cs.SX, cs.SX, cs.SX]:
        assert u.shape[0] == self.nu
        z = x[self.nx-self.nz : self.nx]
        R = cs.SX(cs.vertcat(
            cs.horzcat( 1-2*(x[5]**2+x[6]**2), 2*(x[4]*x[5]-x[3]*x[6]), 2*(x[4]*x[6]+x[3]*x[5]) ),
            cs.horzcat( 2*(x[4]*x[5]+x[3]*x[6]), 1-2*(x[4]**2+x[6]**2), 2*(x[5]*x[6]-x[3]*x[4]) ),
            cs.horzcat( 2*(x[4]*x[6]-x[3]*x[5]), 2*(x[5]*x[6]+x[3]*x[4]), 1-2*(x[4]**2+x[5]**2) ),
        ))
        self.b1 = R[:,0]
        self.b2 = R[:,1]
        self.b3 = R[:,2]

        f = cs.SX(cs.vertcat(
            R@(cs.vertcat(0,0,cs.sum1(u)) - self.A@z[0:3])/self.m + self.g,
            cs.inv(self.J) @ (self.B@u - cs.cross(z[3:6], self.J@z[3:6]))
        ))
        g_m = cs.SX(cs.vertcat(
            self.b3/self.m @ cs.SX.ones(1,self.nu),
            cs.inv(self.J) @ self.B
        ))
        g_um = cs.SX(cs.vertcat(
            cs.horzcat(self.b1, self.b2)/self.m,
            cs.SX.zeros(self.nz-3,self.n_um)
        ))
        return f, g_m, g_um


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

        xdot = cs.SX.sym("xdot", xdot_aug.shape[0])
        f_expl = xdot_aug
        f_impl = xdot - f_expl

        self.x = x_aug
        #self.xdot = xdot
        self.xdot = xdot_aug
        #self.f_expl_expr = f_expl
        #self.f_impl_expr = f_impl
        self.nx, self.nu = self.get_dims()


class PendulumQuadrotor(Quadrotor):
    def __init__(
        self,
        model: Quadrotor,
        m_pend: float,
        l_pend: float,
    ) -> None:
        assert type(model) == Quadrotor
        super().__init__(
            model.m, model.Ixx, model.Iyy, model.Izz,
            model.Ax, model.Ay, model.Az, model.kf, model.km,
            model.xB, model.yB, model.u_min, model.u_max,
            "Pendulum_Quadrotor"
        )
        self.mp = m_pend
        self.lp = l_pend
        self._get_pend_dynamics(mp=m_pend, lp=l_pend)

    def _get_pend_dynamics(
        self,
        mp: float,
        lp: float,
    ) -> None:
        xp = cs.SX.sym("xp")
        yp = cs.SX.sym("yp")
        xp_dot = cs.SX.sym("xp_dot")
        yp_dot = cs.SX.sym("yp_dot")

        zp = cs.sqrt(lp**2 - xp**2 - yp**2)
        x_ddot = self.xdot[-6]
        y_ddot = self.xdot[-5]
        z_ddot = self.xdot[-4]

        F = 1/(zp**2 * (xp**2-lp**2)) * (
            zp**4*y_ddot + yp*zp**3*z_ddot + yp*yp_dot**2*(lp**2-xp**2) + yp*xp_dot**2*(lp**2-yp**2) + 2*xp*yp**2*xp_dot*yp_dot - self.g[2]*yp*zp**3
        )
        xp_ddot_partial = (xp**2-lp**2) / (zp**2*( (xp**2-lp**2)*(yp**2-lp**2)-(xp*yp)**2 )) * (
            xp*zp**3*z_ddot + xp*yp*zp**2*F + xp*xp_dot**2*(lp**2-yp**2) + xp*yp_dot**2*(lp**2-xp**2) + 2*yp*xp**2*xp_dot*yp_dot - self.g[2]*xp*zp**3
        )
        xp_ddot = xp_ddot_partial + (xp**2-lp**2) / (zp**2*( (xp**2-lp**2)*(yp**2-lp**2)-(xp*yp)**2 )) * (zp**4*x_ddot)
        
        yp_ddot_partial = 1/(zp**2 * (xp**2-lp**2)) * (
            yp*zp**3*z_ddot + xp*yp*zp**2*xp_ddot + yp*yp_dot**2*(lp**2-xp**2) + yp*xp_dot**2*(lp**2-yp**2) + 2*xp*yp**2*xp_dot*yp_dot - self.g[2]*yp*zp**3
        )
        yp_ddot = yp_ddot_partial + 1/(zp**2 * (xp**2-lp**2)) * (zp**4*y_ddot)


        acc_due_to_pend = cs.SX((mp/(self.m+mp)) * cs.vertcat(
            -xp_ddot,
            -yp_ddot,
            ((xp*xp_ddot + xp_dot**2 + yp*yp_ddot + yp_dot**2)/zp + (xp*xp_dot + yp*yp_dot)**2/zp**3) + self.g[2]*zp/lp,
        ))
        self.xdot[-6:-3] = self.xdot[-6:-3]*self.m/(self.m+mp) + acc_due_to_pend
        self.xdot = cs.SX(cs.vertcat(self.xdot, xp_dot, yp_dot, xp_ddot, yp_ddot))
        self.x = cs.SX(cs.vertcat(self.x, xp, yp, xp_dot, yp_dot))
        self.nx, self.nu = self.get_dims()


class PendDisturbedQuadrotor(PendulumQuadrotor):
    def __init__(
            self,
            model: PendulumQuadrotor
        ) -> None:
            assert type(model) == PendulumQuadrotor
            super().__init__(
                model=Quadrotor(
                    model.m, model.Ixx, model.Iyy, model.Izz,
                    model.Ax, model.Ay, model.Az, model.kf, model.km,
                    model.xB, model.yB, model.u_min, model.u_max,
                    "Pendulum_Disturbed_Quadrotor"
                ),
                m_pend=model.mp, l_pend=model.lp
            )
            self._get_disturbed_dynamics()

    def _get_disturbed_dynamics(self) -> None:
        d = cs.SX.sym("disturbance", self.nx)
        x_aug = cs.SX(cs.vertcat(self.x, d))

        f_expl_aug = cs.SX(cs.vertcat(
            self.f_expl_expr[:self.nx] + d,
            cs.SX.zeros(self.nx)
        ))
        xdot_aug = cs.SX.sym("xdot_aug", f_expl_aug.shape[0])
        f_impl_aug = xdot_aug - f_expl_aug

        self.x = x_aug
        self.xdot = xdot_aug
        self.f_expl_expr = f_expl_aug
        self.f_impl_expr = f_impl_aug


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

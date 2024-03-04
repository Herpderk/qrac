#!/usr/bin/python3

from qrac.models import Crazyflie, Quadrotor, ParameterAffineQuadrotor
from qrac.trajectory import Circle
from qrac.control.param_nmpc import ParameterAdaptiveNMPC
from qrac.estimation import SetMembershipEstimator, LMS
from qrac.sim.acados_plant import AcadosPlant
from qrac.sim.minimal_sim import MinimalSim
import numpy as np


def main():
    # inaccurate model
    model_inacc = Crazyflie(Ax=0, Ay=0, Az=0)

    # true plant model
    m_true = 1.5 * model_inacc.m
    Ixx_true = 50 * model_inacc.Ixx
    Iyy_true = 50 * model_inacc.Iyy
    Izz_true = 50 * model_inacc.Izz
    Ax_true = 0
    Ay_true = 0
    Az_true = 0
    xB_true = model_inacc.xB
    yB_true = model_inacc.yB
    k_true = model_inacc.k
    u_min_true = model_inacc.u_min
    u_max_true = model_inacc.u_max
    model_acc = Quadrotor(
        m_true, Ixx_true, Iyy_true, Izz_true,
        Ax_true, Ay_true, Az_true, xB_true, yB_true,
        k_true, u_min_true, u_max_true)

    # initialize estimator
    ctrl_T = 0.005
    u_gain = 1000
    p_tol = 0.1*np.ones(10)
    p_min = ParameterAffineQuadrotor(model_acc).get_parameters()\
        - 2*np.abs(ParameterAffineQuadrotor(model_acc).get_parameters())
    p_max = ParameterAffineQuadrotor(model_acc).get_parameters()\
        + 2*np.abs(ParameterAffineQuadrotor(model_acc).get_parameters())
    d_min = -0.1*np.ones(12)
    d_max = -d_min
    
    lms = LMS(
        model=model_inacc, update_gain=u_gain, time_step=ctrl_T
    )
    sm = SetMembershipEstimator(
        model=model_inacc, estimator=lms,
        param_tol=p_tol, param_min=p_min, param_max=p_max,
        disturb_min=d_min, disturb_max=d_max, time_step=ctrl_T,
        qp_tol=10**-6, max_iter=10
    )

    Q = np.diag([1,1,1, 2,2,2, 1,1,1, 2,2,2])
    R = np.diag([0, 0, 0, 0])
    u_min = model_inacc.u_min
    u_max = model_inacc.u_max
    num_nodes = 75
    rti = True
    real_time = True
    sm_mpc = ParameterAdaptiveNMPC(
        model=model_inacc, estimator=sm, Q=Q, R=R,
        u_min=u_min, u_max=u_max, time_step=ctrl_T,
        num_nodes=num_nodes, rti=rti, real_time=real_time,
        nlp_tol=10**-6, nlp_max_iter=1, qp_max_iter=5
    )

    # initialize simulator plant
    sim_T = ctrl_T / 10
    plant = AcadosPlant(
        model=model_acc, sim_step=sim_T, control_step=ctrl_T)

    # initialize simulator
    lb_pose = [-10, -10, 0]
    ub_pose = [10, 10, 10]
    sim = MinimalSim(
        plant=plant, controller=sm_mpc,
        lb_pose=lb_pose, ub_pose=ub_pose,
        data_len=2000, real_time=real_time)

     # define a circular trajectory
    traj = Circle(v=4, r=8, alt=8)

    # Run the sim for N control loops
    x0 = np.array([8,0,0, 0,0,0, 0,0,0, 0,0,0])
    N = int(round(30 / ctrl_T))      # 30 seconds worth of control loops
    sim.start(x0=x0, max_steps=N, verbose=True)

    # track the given trajectory
    xset = np.zeros(sm_mpc.n_set)
    nx = model_inacc.nx
    dt = sm_mpc.dt
    t0 = sim.timestamp

    sm_mpc.start()
    while sim.is_alive:
        t = sim.timestamp
        for k in range(num_nodes):
            xset[k*nx : k*nx + nx] = \
                np.array(traj.get_setpoint(t - t0))
            t += dt
        sim.update_setpoint(xset=xset)
    sm_mpc.stop()
    
    print("acc params:")
    print(  
        ParameterAffineQuadrotor(model_acc).get_parameters()
    )


if __name__=="__main__":
    main()
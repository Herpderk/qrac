#!/usr/bin/python3

from qrac.models import Crazyflie, Quadrotor, AffineQuadrotor
from qrac.trajectory import Circle
from qrac.control.param_nmpc import ParameterAdaptiveNMPC
from qrac.estimation import SetMembershipEstimator, MHE
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

    # init estimator
    Q_mhe = 1*np.diag([1,1,1,1,1,1,1,1,1,1])
    R_mhe = 1000 * np.diag([1,1,1, 1,1,1, 1,1,1, 1,1,1])
    p_min = AffineQuadrotor(model_acc).get_parameters()\
        - 2*np.abs(AffineQuadrotor(model_acc).get_parameters())
    p_max = AffineQuadrotor(model_acc).get_parameters()\
        + 2*np.abs(AffineQuadrotor(model_acc).get_parameters())
    d_min = -0.1*np.ones(12)
    d_max = -d_min
    ctrl_T = 0.01
    num_nodes_mhe = 20
    mhe = MHE(
        model=model_inacc, Q=Q_mhe, R=R_mhe,
        param_min=p_min, param_max=p_max,
        disturb_min=d_min, disturb_max=d_max,
        time_step=ctrl_T, num_nodes=num_nodes_mhe,
        rti=True, nonlinear=False,
        nlp_tol=10**-6, nlp_max_iter=1, qp_max_iter=3
    )

    p_tol = 0.1*np.ones(10)
    sm = SetMembershipEstimator(
        model=model_inacc, estimator=mhe,
        param_tol=p_tol, param_min=p_min, param_max=p_max,
        disturb_min=d_min, disturb_max=d_max, time_step=ctrl_T,
        qp_tol=10**-6, max_iter=10
    )

    # init mpc
    Q = np.diag([1,1,1, 2,2,2, 1,1,1, 2,2,2])
    R = np.diag([0, 0, 0, 0])
    u_min = model_inacc.u_min
    u_max = model_inacc.u_max
    num_nodes = 75
    real_time = False
    mhe_mpc = ParameterAdaptiveNMPC(
        model=model_inacc, estimator=sm, Q=Q, R=R,
        u_min=u_min, u_max=u_max, time_step=ctrl_T,
        num_nodes=num_nodes, real_time=real_time, rti=True, 
        nlp_tol=10**-6, nlp_max_iter=1, qp_max_iter=5
    )

    # init simulator plant
    sim_T = ctrl_T / 10
    plant = AcadosPlant(
        model=model_acc, sim_step=sim_T, control_step=ctrl_T)

    # init simulator
    lb_pose = [-10, -10, 0]
    ub_pose = [10, 10, 10]
    sim = MinimalSim(
        plant=plant, controller=mhe_mpc,
        lb_pose=lb_pose, ub_pose=ub_pose,
        data_len=2000, real_time=real_time)

     # define a circular trajectory
    traj = Circle(v=4, r=8, alt=8)

    # Run the sim for N control loops
    x0 = np.array([8,0,0, 0,0,0, 0,0,0, 0,0,0])
    N = int(round(30 / ctrl_T))      # 30 seconds worth of control loops
    sim.start(x0=x0, max_steps=N, verbose=True)

    # track the given trajectory
    xset = np.zeros(mhe_mpc.n_set)
    nx = model_inacc.nx
    dt = mhe_mpc.dt
    t0 = sim.timestamp

    #mhe_mpc.start()
    while sim.is_alive:
        t = sim.timestamp
        for k in range(num_nodes):
            xset[k*nx : k*nx + nx] = \
                np.array(traj.get_setpoint(t - t0))
            t += dt
        sim.update_setpoint(xset=xset)
    #mhe_mpc.stop()
    
    print("acc params:")
    print(  
        AffineQuadrotor(model_acc).get_parameters()
    )



if __name__=="__main__":
    main()

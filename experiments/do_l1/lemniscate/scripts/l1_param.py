#!/usr/bin/python3

import numpy as np
from qrac.models import Quadrotor, Crazyflie
from qrac.control import NMPC
from qrac.control import L1Augmentation
from qrac.sim import MinimalSim
from consts import A_GAINS


def run_sim(
    A_GAIN: float,
    i: int,
    M: int,
    acc: Quadrotor,
    inacc: Quadrotor
):
    # mpc settings
    CTRL_T = 0.01
    NODES = 50
    Q = np.diag([1,1,1, 1,1,1,1, 1,1,1, 1,1,1,])
    R = np.diag([0, 0, 0, 0])

    # L1 settings
    W = 1000

    # L1 optimizer settings
    A_MIN = 0
    A_MAX = 10**4.5
    Q_l1 = 10**-7

    # sim settings
    SIM_T = CTRL_T / 10

    # file access
    xfilename = "../refs/xref.npy"
    ufilename = "../refs/uref.npy"
    dfilename = "../refs/disturb.npy"

    # load in time optimal trajectory
    xref = np.load(xfilename)
    uref = np.load(ufilename)
    disturb = np.load(dfilename)

    # init mpc
    mpc = NMPC(
        model=inacc, Q=Q, R=R,
        u_min=inacc.u_min, u_max=inacc.u_max,
        time_step=CTRL_T, num_nodes=NODES,
        rti=True, nlp_max_iter=1, qp_max_iter=10
    )

    # init L1
    if M:
        l1_mpc = L1Augmentation(
            model=inacc, control_ref=mpc,
            adapt_gain=A_GAIN, bandwidth=W,
            adapt_gain_min=A_MIN, adapt_gain_max=A_MAX, Q=Q_l1
        )
    else:
        l1_mpc = L1Augmentation(
            model=inacc, control_ref=mpc,
            adapt_gain=A_GAIN, bandwidth=W
        )

    # init sim
    steps = xref.shape[0]
    sim = MinimalSim(
        model=acc, data_len=steps,
        sim_step=SIM_T, control_step=CTRL_T,
    )

    # run for predefined number of steps
    nx = inacc.nx
    xset = np.zeros(nx*NODES)
    x = xref[0]
    nu = inacc.nu
    uset = np.zeros(nu*NODES)
    a_gains = np.zeros(steps)

    for k in range(steps - NODES):
        diff = steps - k
        if diff < NODES:
            xset[:nx*diff] = xref[k : k+diff, :].flatten()
            xset[:nx*(NODES-diff)] = np.tile(xref[-1,:], NODES-diff)
            uset[:nu*diff] = uref[k : k+diff, :].flatten()
            uset[:nu*(NODES-diff)] = np.tile(uref[-1,:], NODES-diff)

        else:
            xset[:] = xref[k:k+NODES, :].flatten()
            uset[:nu*NODES] = uref[k:k+NODES, :].flatten()

        u = l1_mpc.get_input(x=x, xset=xset, uset=uset, timer=True)
        a_gains[k] = l1_mpc._a_gain
        
        d = 0*np.hstack(
            (np.zeros(9), disturb[k,-4:])
        )
        x = sim.update(x=x, u=u, d=d, timer=True)

        print(f"\nu: {u}")
        print(f"x: {x}")
        print(f"sim time: {(k+1)*CTRL_T}\n")

    # calculate RMSE
    xdata = sim.get_xdata()
    err = np.sum(( xref[:,0:3] - xdata[:,0:3] )**2,axis=-1)**0.5
    rmse = np.sqrt(np.sum(np.square(err))/err.shape[0])
    print(f"root mean square error: {rmse}")

    # save bandwidth history
    np.save(f"../data/gains_M={M}_A={A_GAIN}_I={i}", a_gains)
    np.save(f"../data/traj_M={M}_A={A_GAIN}_I={i}", xdata)

def main():
    UNCERTAINTY = 0.4
    SAMPLE = 200
    
    # inaccurate model
    inacc = Crazyflie(Ax=0.1, Ay=0.1, Az=0.2)
    p_inacc = np.hstack((
        inacc.m, inacc.Ixx, inacc.Iyy, inacc.Izz,
        inacc.Ax, inacc.Ay, inacc.Az,
        inacc.xB, inacc.yB,
    ))
    
    for A_GAIN in A_GAINS:
        for i in range(SAMPLE):
            p_acc = p_inacc + UNCERTAINTY * p_inacc * np.random.uniform(-1,1, p_inacc.shape[0])
            acc = Quadrotor(
                m=p_acc[0], Ixx=p_acc[1],Iyy=p_acc[2], Izz=p_acc[3],
                Ax=p_acc[4], Ay=p_acc[5], Az=p_acc[6], kf=inacc.kf, km=inacc.km,
                xB=p_acc[7:11], yB=p_acc[11:15], u_min=inacc.u_min, u_max=inacc.u_max
            )
            run_sim(A_GAIN=A_GAIN, i=i, M=0, acc=acc, inacc=inacc)
            run_sim(A_GAIN=A_GAIN, i=i, M=1, acc=acc, inacc=inacc)

if __name__=="__main__":
    main()

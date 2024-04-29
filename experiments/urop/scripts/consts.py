#!/usr/bin/python3

import numpy as np


# relative file paths
LEM_XREF_PATH = "../data/lem_xref.npy"
LEM_UREF_PATH = "../data/lem_uref.npy"
LEM_D_PATH = "../data/lem_d.npy"

CIRC_XREF_PATH = "../data/circ_xref.npy"
LEM_UREF_PATH = "../data/circ_uref.npy"
LEM_D_PATH = "../data/circ_d.npy"

SETPT_XREF_PATH = "../data/setpt_xref.npy"
SETPT_UREF_PATH = "../data/setpt_uref.npy"
SETPT_D_PATH = "../data/setpt_d.npy"



# trajectory planner settings
PREDICT_TIME = 30
CTRL_T = 0.01
Q_TRAJ = np.diag([20,20,20, 1,1,1,1, 1,1,1, 1,1,1,])
R_TRAJ = np.diag([0, 0, 0, 0])


# mpc settings
NODES = 40
MAX_ITER_NMPC = 10
Q = np.diag([4,4,4, 2,2,2,2, 1,1,1, 1,1,1,])
R = np.diag([0, 0, 0, 0])


# mhe settings
Q_MHE_LIN = 1 * np.diag([
    1, 1,1,1, 1,1,1, 1,1,1
])
Q_MHE_NONLIN = 1 * np.diag([
    1, 1,1,1, 1,1,1,
])
R_MHE = 1 * np.diag([
    1,1,1, 1,1,1,1, 1,1,1, 1,1,1
])
NODES_MHE = 20
MAX_ITER_MHE = 5


# lms settings
U_GAIN = 1000


# set-membership settings
P_TOL = np.array([
    0.2, 0.02,0.02,0.02, 2000,2000,2000, 0.02,0.02,0.02
])
MAX_ITER_SM = 2


# L1 settings
A_GAIN = 20
W = 100


# sim settings
SIM_T = CTRL_T / 10
D_MAX = np.array([
    0,0,0, 0,0,0,0, 4,4,4, 4,4,4,
])
D_MIN = -D_MAX



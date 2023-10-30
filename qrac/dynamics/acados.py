#!/usr/bin/python3

from acados_template import AcadosModel
from qrac.dynamics.casadi import CasadiModel


def get_acados_model(model_cs: CasadiModel) -> AcadosModel:
    model = AcadosModel()
    model.f_expl_expr = model_cs.f_expl_expr
    model.x = model_cs.x
    model.xdot = model_cs.xdot
    model.u = model_cs.u
    model.name = model_cs.name
    return model

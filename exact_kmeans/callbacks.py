import logging

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from exact_kmeans.util import print_variables

logger = logging.getLogger(__name__)


def add_cuts(model: gp.Model, where: int, v: int, n: int, k: int) -> None:
    if where == GRB.Callback.MIPNODE:
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status != GRB.OPTIMAL:
            return
        for i in range(v, n):
            for j in range(i + 1, n):
                for ll in range(v, k):
                    zs = [
                        model.getVarByName(f"z[{i},{j},{ll},{m}]") for m in range(v, n)
                    ]
                    relaxed_z = model.cbGetNodeRel(zs)  # type: ignore
                    y = model.getVarByName(f"y[{i},{j},{ll}]")
                    if not np.isclose(
                        sum(relaxed_z), model.cbGetNodeRel(y), atol=0.001  # type: ignore
                    ):
                        logger.info(
                            f"Add cut for z[{i},{j},{ll},:] ({sum(relaxed_z)}) "
                            "and y[{i},{j},{ll}] ({model.cbGetNodeRel(y)})"
                        )
                        model.cbCut(gp.quicksum(zs) == y)  # type: ignore


def variable_bounds(model: gp.Model, where: int) -> None:
    if where in {GRB.Callback.MIPNODE, GRB.Callback.MIPSOL, GRB.Callback.MIP}:
        for var in model.getVars():
            # This does not work, the lower and upper bounds of the variables are never updated?
            if var.LB != 0 or var.UB != 1:
                print()
                print(var, var.LB, var.UB)
                print()


def relax_callback(model: gp.Model, where: int) -> None:
    if where == GRB.Callback.MIPNODE:
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status == GRB.OPTIMAL:
            with open("relaxed-example.txt", "w") as of:
                for txt in print_variables(
                    model.getVars(), model.cbGetNodeRel(model.getVars())
                ):
                    of.write(txt)
                # dd = {
                #     var.VarName: val
                #     for var, val in zip(
                #         model.getVars(), model.cbGetNodeRel(model.getVars())
                #     )
                # }
            model.terminate()

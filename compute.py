"""
"""

import os

from oemof.solph import EnergySystem, Model

# DONT REMOVE THIS LINE!
from oemof.tabular import datapackage  # noqa
from oemof.tabular.facades import TYPEMAP
import oemof.tabular.tools.postprocessing as pp
import oemof.outputlib as outputlib

import pyomo.core as po

name = "1-node"

# path to directory with datapackage to load
datapackage_dir = "datapackages/1-node/"

# create  path for results (we use the datapackage name to store results)
results_path = os.path.join(
    "results", name, "output")
if not os.path.exists(results_path):
    os.makedirs(results_path)

# create energy system object
es = EnergySystem.from_datapackage(
    os.path.join(datapackage_dir, "datapackage.json"),
    attributemap={},
    typemap=TYPEMAP,
)

# create model from energy system
m = Model(es)

# select solver 'gurobi', 'cplex', 'glpk' etc
m.solve("gurobi")

# get the results from the the solved model
m.results = m.results()

pp.write_results(m, results_path)

modelstats = outputlib.processing.meta_results(m)

# calculate fix costs
# https://github.com/oemof/oemof/pull/396
fix_cost = sum(
    [n.capacity * n.fix_cost
        for n in es.nodes if all(
            hasattr(n, attr) for attr in ['fix_cost', 'capacity'])])

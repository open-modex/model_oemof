"""
"""

import os

from oemof.solph import EnergySystem, Model

# DONT REMOVE THIS LINE!
from oemof.tabular import datapackage  # noqa
from oemof.tabular.facades import TYPEMAP
import oemof.tabular.tools.postprocessing as pp
import oemof.outputlib as outputlib

import numpy as np

import plotly.graph_objs as go
import plotly.offline as offline

# path to directory with datapackage to load
path = "datapackages/"
packages = os.listdir(path)
meta_results = {}

for pk in packages:

    # create a results directory
    results_path = os.path.join(
        "results", pk, "output")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # create energy system object
    es = EnergySystem.from_datapackage(
        os.path.join(path, pk, "datapackage.json"),
        attributemap={},
        typemap=TYPEMAP,
    )

    # collect fix costs
    # https://github.com/oemof/oemof/pull/396
    fix_cost = sum(
        [n.capacity * n.fix_cost
            for n in es.nodes if all(
                hasattr(n, attr) for attr in ['fix_cost', 'capacity'])])

    # create model from energy system
    m = Model(es)

    # select solver 'gurobi', 'cplex', 'glpk' etc
    m.solve("gurobi")

    # get the results from the the solved model
    m.results = m.results()

    pp.write_results(m, results_path)

    # store model statistics
    meta_results[pk] = outputlib.processing.meta_results(m)
    meta_results[pk]['obj_add_fix_cost'] = meta_results[pk]['objective'] + fix_cost


# view meta results
columns = [
    'Lower bound', 'Upper bound', 'Number of constraints',
    'Number of variables', 'Number of nonzeros']

values = [
    [name, v['objective'], v['obj_add_fix_cost']]
     + [v['problem'][c] for c in columns]
    for name, v in meta_results.items()]

fig = go.Figure(data=[go.Table(
    header=dict(values=['Name', 'Objective', 'With fixed cost'] + columns),
    cells=dict(values=np.array(values).T))])

fig.update_layout(
    height=800,
    title_text='Meta results - For more detailed results look at the results folder.'
)

offline.plot(fig, filename='results.html', auto_open=True)

"""
"""

import os
import time
import urllib

from oemof.solph import EnergySystem, Model
from oemof.tabular.tools.postprocessing import supply_results

# DONT REMOVE THIS LINE!
from oemof.tabular import datapackage  # noqa
from oemof.tabular.facades import TYPEMAP
import oemof.tabular.tools.postprocessing as pp
import oemof.outputlib as outputlib
from oemof.network import Bus

import numpy as np
import pandas as pd

import plotly.graph_objs as go
import plotly.offline as offline

from markdown import markdown
from bs4 import BeautifulSoup

# Get open_MODEX/results/README.md markdown mapping table
res = urllib.request.urlopen("https://raw.githubusercontent.com/open-modex/results/master/README.md")
md = markdown(res.read().decode('utf-8'), extensions=['tables'])
tables = BeautifulSoup(md, features="html.parser").findAll('table')
rows = lambda t: t.find('tbody').findAll('tr')
cells = lambda r: list(c.text for c in r.findAll('td'))
mapping, filenames = [pd.DataFrame([cells(r) for r in rows(t)]) for t in tables]
mapping.columns = ['id', 'name', 'description']
mapping.set_index('name', inplace=True)

# path to directory with datapackage to load
path = "datapackages/"
packages = os.listdir(path)
meta_results = {}

for pk in packages:

    ni = mapping.loc[pk, 'id']  # numeric identifier

    # create a results directory
    results_path = os.path.join(
        "results", "oemof", ni)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        os.makedirs(os.path.join(results_path, "output"))

    build_time = time.time()

    # create energy system object
    es = EnergySystem.from_datapackage(
        os.path.join(path, pk, "datapackage.json"),
        attributemap={},
        typemap=TYPEMAP,
    )

    build_time = time.time() - build_time

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

    # store model statistics
    meta_results[pk] = outputlib.processing.meta_results(m)
    meta_results[pk]['obj_add_fix_cost'] = meta_results[pk]['objective'] + fix_cost

    pp.write_results(m, os.path.join(results_path, "output"))

    # POC write results in modex format
    results = {
        "timebuild_oemof.csv":
            "Time required to build the model,Seconds,%s\n" % build_time,
        "constraints_oemof.csv":
            "Number of constraints,#,%s\n" % meta_results[pk]['problem']['Number of constraints'],
        "variables_omeof.csv":
            "Number of variables,#,%s\n" % meta_results[pk]['problem']['Number of variables'],
        "objective_omeof.csv":
            "Objective value,EURO,%s\n" % meta_results[pk]['obj_add_fix_cost']
    }

    buses = [b.label for b in m.es.nodes if isinstance(b, Bus)]

    production = []
    for b in buses:
        production.append(supply_results(results=m.results, es=m.es, bus=[b]))

    pd.concat(production, axis=1).\
        to_csv(os.path.join(results_path, "production_oemof" + ".csv"))

    for fn, content in results.items():
        with open(os.path.join(results_path, fn), "w") as f:
            f.write(content)

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

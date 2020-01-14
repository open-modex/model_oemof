### "imports"

import os

from oemof.tabular.facades import Load, Dispatchable, Volatile, Link, Bus
import oemof.solph as solph
import oemof.tabular.datapackage
import oemof.tabular.datapackage.building as dp
import pandas as pd


### "infer-metadata"
dp.infer_metadata(
    path='datapackage/',
    package_name="minimal-oemof-tabular-example",
    foreign_keys={
        "bus": ["volatile", "load", "dispatchable"],
        "profile": ["load"],
        "from_to_bus": ["link"],
    },
)

### "create-es"
es = solph.EnergySystem.from_datapackage(
    "datapackage/datapackage.json",
    typemap={
        "bus": Bus,
        "dispatchable": Dispatchable,
        "sink": Load,
        "link": Link,
        "volatile": Volatile
    },
)

### "solve"

om = solph.Model(es)

om.solve(solver="gurobi")

### "Results"

from pprint import pprint as pp

import oemof.outputlib.processing as process


results = process.results(om)
results = {
    (s.label, t.label): list(
        results[(s, t)]["sequences"]["flow"]
    )
    for (s, t) in results
}
pp(results)

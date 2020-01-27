### "imports"

import os

from oemof.tabular.facades import Load, Dispatchable, Volatile, Link, Bus, Excess, Shortage
import oemof.solph as solph
import oemof.tabular.datapackage
import oemof.tabular.datapackage.building as dp
import pandas as pd


### "infer-metadata"
dp.infer_metadata(
    path='datapackage/',
    package_name="minimal-oemof-tabular-example",
    foreign_keys={
        "bus": ["volatile", "load", "dispatchable", "excess", "shortage"],
        "profile": ["load", "volatile"],
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
        "volatile": Volatile,
        "excess": Excess,
        "shortage": Shortage
    },
)

### "add-timeindex"
# Needed on Windows OS
es.timeindex = pd.date_range(
    "2016-01-01T00:00:00+0100", periods=8784, freq="60min"
)

### "solve"

om = solph.Model(es)

om.solve(solver="glpk")

### "Results"
import oemof.outputlib.processing as process

print(process.meta_results(om))

#from pprint import pprint as pp
#results = process.results(om)
#
#results = {
#    (s.label, t.label): list(
#        results[(s, t)]["sequences"]["flow"]
#    )
#    for (s, t) in results
#}
#pp(results)
#
#import oemof.outputlib.views as views
#import matplotlib.pyplot as plt
#
#results = views.convert_keys_to_strings(results)
#results[('BB', 'bb-excess')]['sequences'].plot()
#plt.show()

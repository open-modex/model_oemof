from collections import namedtuple
from csv import DictWriter, field_size_limit as csv_field_size_limit
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from functools import reduce
from itertools import chain, groupby
from pathlib import Path
from pprint import pformat as pf
from typing import Tuple
import json
import sys
import textwrap

from frictionless import Detector, Package, Resource
from loguru import logger
from oemof.solph import (
    Bus,
    EnergySystem as ES,
    Flow,
    GenericStorage as Storage,
    Investment,
    Model,
    Sink,
    Source,
    Transformer,
    processing,
)
from oemof.tools.economics import annuity
import pandas as pd


csv_field_size_limit(sys.maxsize)

logger.disable(__name__)

DE = [
    "BB",
    "BE",
    "BW",
    "BY",
    "HB",
    "HE",
    "HH",
    "MV",
    "NI",
    "NW",
    "RP",
    "SH",
    "SL",
    "SN",
    "ST",
    "TH",
]


def slurp(path):
    with open(path) as f:
        return json.load(f)


def invest(mapping):
    if mapping[0].year == 2016 or (
        "capital costs" not in mapping[1]
        and "expansion limit" not in mapping[1]
    ):
        return {
            "nominal_storage_capacity"
            if mapping[0].technology[0] == "storage"
            else "nominal_value": mapping[1].get("installed capacity", 0)
            * mapping[1].get("E2P ratio", 1)
        }
    optionals = (
        {"maximum": mapping[1]["expansion limit"]}
        if "expansion limit" in mapping[1]
        and mapping[1]["expansion limit"] != 999999.0
        else {}
    )
    if "E2P ratio" in mapping[1]:
        ratio = 1 / mapping[1]["E2P ratio"]
        optionals["invest_relation_input_capacity"] = ratio
        optionals["invest_relation_output_capacity"] = ratio
    return {
        "investment": Investment(
            ep_costs=annuity(
                mapping[1].get(
                    "capital costs",
                    # TODO: Retrieve he capital costs of 446.39 for
                    #       transmission lines from the data instead of
                    #       hardcoding it here.
                    0 if len(mapping[0].regions) == 1 else 446.39,
                )
                * mapping[1].get("distance", 1)
                / (
                    # For some reason, all storage capital costs (CCs) are in
                    # €/MW, except for for batteries, which are in €/MWh.
                    # Therefore the CCs for every non-battery storage have to
                    # be divided by the E2P ratio, which is in hours, in order
                    # to account for the fact that our storage capacities are
                    # in MWh.
                    mapping[1].get("E2P ratio", 1)
                    if mapping[0].technology[1] != "battery"
                    else 1
                ),
                # TODO: Retrieve the lifetime of 40 years for transmissions
                #       lines from the data instead of hardcoding it. Same with
                #       the WACC of 0.07 in the line below.
                mapping[1]["lifetime"] if len(mapping[0].regions) == 1 else 40,
                0.07,
            )
            if "capital costs" in mapping[1] or len(mapping[0].regions) == 2
            else 0,
            existing=mapping[1].get("installed capacity", 0)
            * mapping[1].get("E2P ratio", 1),
            **optionals,
        )
    }


@dataclass(eq=True, frozen=True)
class Key(dict):
    region: Tuple[str]
    technology: Tuple[str, str]
    vectors: Tuple[str, str]
    year: int

    @classmethod
    def from_dictionary(cls, d):
        # Repair the "NS" region. This one seems to be a typo.
        # Should be "NI" instead.
        if "NS" in d["region"]:
            d["region"] = [("NI" if r == "NS" else r) for r in d["region"]]
        arguments = dict(
            region=tuple(sorted(d["region"])),
            technology=(d["technology"], d["technology_type"]),
            vectors=(d["input_energy_vector"], d["output_energy_vector"]),
            year=(
                d["year"]
                if "year" in d
                else pd.to_datetime(str(d["timeindex_start"])).year
            ),
        )
        return cls(**arguments)

    @property
    def regions(self):
        """`regions` is an alias of `region`.

        `region` is the key used in the original JSON data, but since it's a
        tuple, even it mostly with just one entry, `regions` is a less
        confusing name.
        """
        return self.region

    def __post_init__(self):
        names = [field.name for field in fields(self)]
        for name in names:
            self[name] = getattr(self, name)


def reducer(dictionary, value):
    key = Key.from_dictionary(value)
    # Expand parameters which are applied to multiple regions by having more
    # than two regions in the "region" key (two regions would mean
    # ["source", "target"] instead), to a list copies of the same parameter
    # with only one "region".
    keys = (
        [replace(key, region=(kr,)) for kr in key.region]
        if len(key.region) > 2
        else [key]
    )
    for key in keys:
        if key in dictionary:
            assert (value["parameter_name"] not in dictionary[key]) or (
                dictionary[key][value["parameter_name"]]
                == (value["value"] if "value" in value else value["series"])
            ), textwrap.indent(
                f'\n\nParameter\n\n  {value["parameter_name"]}\n'
                f"\nalready present under\n\n  {key}\n"
                f'\nOld value: {dictionary[key][value["parameter_name"]]}'
                f'\nNew value: {value.get("value", "Series ommitted...")}',
                "  ",
            )
        else:
            dictionary[key] = {}
        dictionary[key][value["parameter_name"]] = (
            value["value"] if "value" in value else value["series"]
        )
    logger.info(value)
    return dictionary


def from_json(path="base-scenario.concrete.json"):
    base = {"concrete": slurp(path)}
    for mapping in base:
        logger.info(
            f"\n{mapping} top-level keys/lengths:"
            f"\n{pf([(k, len(base[mapping][k])) for k in base[mapping]])}"
        )
    # Time series boundaries
    tsbs = {
        mapping: set(
            [
                (ts["timeindex_start"], ts["timeindex_stop"])
                for ts in base[mapping]["oed_timeseries"]
            ]
        )
        for mapping in base
    }
    for mapping in base:
        logger.info(f"\n{mapping} time series boundaries:" f"\n{pf(tsbs)}")
    reduced = reduce(
        reducer,
        chain(
            base["concrete"]["oed_scalars"], base["concrete"]["oed_timeseries"]
        ),
        {},
    )
    result = {
        mapping: {
            "base": base[mapping],
            "timeseries boundaries": tsbs[mapping],
        }
        for mapping in base
    }
    result["concrete"]["objects"] = reduced
    years = [2016, 2030, 2050]
    result.update(
        {
            year: {k: reduced[k] for k in reduced if k.year == year}
            for year in years
        }
    )
    result["techs"] = {year: techs(result[year]) for year in years}
    result["vectors"] = {year: vectors(result[year]) for year in years}
    # assertion: len(o.region) == 2
    # =>  o.technology == ('transmission', 'hvac')
    # and o.vectors    == ('electricity', 'electricity')
    #
    # len(o.region) > 2 => 16 (DE) or 18
    return result


def techs(mappings):
    return set([m.technology for m in mappings])


def vectors(mappings):
    return set([m.vectors for m in mappings])


Label = namedtuple("Label", ["regions", "technology", "vectors", "name"])


def label(mapping, name):
    return Label(
        mapping[0].regions, mapping[0].technology, mapping[0].vectors, name
    )


def demands(mappings, buses):
    return [
        Sink(
            label=label(demand, "demand"),
            inputs={
                buses[(demand[0].regions[0], demand[0].vectors[0])]: Flow(
                    fix=demand[1]["demand"], nominal_value=1
                )
            },
        )
        for demand in find(mappings, "demand")
    ]


def transmission(line, buses, ratios):
    loss_bus = Bus(label=label(line, "losses"))
    loss = Sink(label=label(line, "loss-sink"), inputs={loss_bus: Flow()})
    flow_bus = Bus(label=label(line, "flow-bus"))
    flow = Sink(
        label=label(line, "energy flow (both directions)"),
        inputs={flow_bus: Flow(min=0, **invest(line))},
    )

    lines = [
        Transformer(
            label=label(line, "energy flow")._replace(regions=regions),
            inputs={source: Flow()},
            outputs={flow_bus: Flow(), loss_bus: Flow(), target: Flow()},
            conversion_factors={
                flow_bus: ratios[regions[0]]["ir"],
                loss_bus: 1 - ratios[regions[1]]["or"],
                source: ratios[regions[0]]["ir"],
                target: ratios[regions[1]]["or"],
            },
        )
        for regions in [line[0].regions, tuple(reversed(line[0].regions))]
        for source in [buses[(regions[0], line[0].vectors[0])]]
        for target in [buses[(regions[1], line[0].vectors[1])]]
    ]
    return lines + [flow_bus, flow, loss_bus, loss]


def lines(mappings, buses):
    ratios = {
        ratio[0].regions[0]: {
            "ir": ratio[1]["input ratio"],
            "or": ratio[1]["output ratio"],
        }
        for ratio in find(
            mappings,
            "input ratio",
            "output ratio",
            technology=("transmission", "hvac"),
            vectors=("electricity", "electricity"),
        )
        + find(
            mappings,
            "input ratio",
            "output ratio",
            technology=("transmission", "DC"),
            vectors=("electricity", "electricity"),
        )
    }
    return [
        node
        for line in find(mappings, technology=("transmission", "hvac"))
        + find(mappings, technology=("transmission", "DC"))
        if len(line[0].regions) == 2
        for node in transmission(line, buses, ratios)
    ]


def trades(mappings, buses):
    imports = find(mappings, technology=("transmission", "trade import"))
    exports = find(mappings, technology=("transmission", "trade export"))
    sinks = [
        Sink(
            label=label(trade, "export"),
            inputs={
                buses[(trade[0].regions[0], trade[0].vectors[0])]: Flow(
                    fix=trade[1]["trade volume"], nominal_value=1
                )
            },
        )
        for trade in exports
    ]
    sources = [
        Source(
            label=label(trade, "import"),
            outputs={
                buses[(trade[0].regions[0], trade[0].vectors[0])]: Flow(
                    fix=trade[1]["trade volume"],
                    nominal_value=trade[1]["installed capacity"],
                )
            },
        )
        for trade in imports
    ]
    return sinks + sources


def fixed(mappings, buses):
    sources = [
        Source(
            label=label(source, "electricity generation"),
            outputs={
                transformer: Flow(
                    fix=source[1]["capacity factor"],
                    **invest(source),
                    variable_costs=source[1].get("variable costs", 0),
                )
            },
        )
        for source in find(mappings, "capacity factor")
        for renewables in [buses[("DE", "renewable share")]]
        for transformer in [
            Transformer(
                label=label(source, "splitter",),
                outputs={
                    buses[
                        (source[0].regions[0], source[0].vectors[1])
                    ]: Flow(),
                    renewables: Flow(),
                    **(
                        {buses[(source[0].regions, "photovoltaics")]: Flow()}
                        if source[0].technology[0] == "photovoltaics"
                        and (source[0].regions, "photovoltaics") in buses
                        else {}
                    ),
                },
                conversion_factors={renewables: renewables.share - 1},
            )
        ]
    ]
    return sources + [o for source in sources for o in source.outputs]


def flexible(mappings, buses):
    limits = {
        (l[0].regions, l[0].vectors[0]): (
            l[1]["natural domestic limit"] * (pow(10, 9) / 3600)
        )
        for l in find(mappings, "natural domestic limit")
    }
    fueled = chain(
        find(mappings, "emission factor"),
        find(mappings, technology=("geothermal", "unknown"),),
    )
    co2c = find(mappings, vectors=("unknown", "co2"))[0][1]["emission costs"]
    sources = [
        Source(
            label=label(f, "electricity generation"),
            outputs={
                source_bus: Flow(
                    **invest(f),
                    variable_costs=(
                        f[1]["variable costs"]
                        + (1 / f[1]["output ratio"])
                        * (
                            f[1].get("emission factor", 0) * co2c
                            + f[1].get("fuel costs", 0)
                        )
                    ),
                    **(
                        {
                            "summed_max": (
                                limits[(f[0].regions, f[0].vectors[0])]
                                / f[1]["installed capacity"]
                            )
                        }
                        if (f[0].regions, f[0].vectors[0]) in limits
                        else {}
                    ),
                )
            },
        )
        for f in fueled
        for source_bus in [Bus(label=label(f, "auxiliary-bus"))]
        for renewables in [buses[("DE", "renewable share")]]
        for transformer in [
            Transformer(
                label=label(f, "auxiliary-transformer"),
                inputs={source_bus: Flow()},
                outputs={
                    buses[(f[0].regions[0], f[0].vectors[1])]: Flow(),
                    buses[("DE", "co2")]: Flow(),
                    renewables: Flow(),
                    **(
                        {buses[("DE", "waste")]: Flow()}
                        if "waste" in f[0].vectors
                        else {}
                    ),
                },
                conversion_factors={
                    buses[("DE", "co2")]: 1
                    * f[1].get("emission factor", 0)
                    / f[1]["output ratio"],
                    renewables: renewables.share
                    - (0 if "geothermal" not in f[0].technology else 1),
                    **(
                        {buses[("DE", "waste")]: 1 / f[1]["output ratio"]}
                        if "waste" in f[0].vectors
                        else {}
                    ),
                },
            )
        ]
    ]
    return (
        sources
        + [b for source in sources for b in source.outputs]
        + [t for source in sources for b in source.outputs for t in b.outputs]
    )


def storages(mappings, buses):
    return [
        Storage(
            label=label(storage, "storage"),
            **investment,
            initial_storage_level=0,
            inflow_conversion_factor=storage[1]["input ratio"],
            outflow_conversion_factor=storage[1]["output ratio"],
            inputs={
                buses[(storage[0].regions[0], storage[0].vectors[0])]: Flow(
                    **nv, variable_costs=storage[1].get("variable costs", 0),
                )
            },
            outputs={
                buses[(storage[0].regions[0], storage[0].vectors[1])]: Flow(
                    **nv, variable_costs=0,
                )
            },
        )
        for storage in find(mappings, "E2P ratio")
        for investment in [invest(storage)]
        for nv in [
            {"nominal_value": storage[1].get("installed capacity", 0)}
            if "investment" not in investment
            else {
                # The investment is bounded by the storage capacity through the
                # flow conversion factors, so we can keep ep_costs and maximum
                # at the default valus of `0` and `inf` respectively.
                "investment": Investment(
                    existing=storage[1].get("installed capacity", 0)
                )
            }
        ]
    ]


def build(mappings, year):
    es = ES(
        timeindex=pd.date_range(
            f"{year}-01-01T00:00:00", f"{year}-12-31T23:00:00", freq="1h"
        )
    )

    rvs = set(
        (region, vector)
        for m in mappings
        for region in m.region
        for vector in m.vectors
        if not vector == "unknown"
    )
    buses = {rv: Bus(label=rv) for rv in rvs}
    pv_regions = {
        r
        for m in mappings
        for r in m.region
        if r in DE
        if Key(
            (r,),
            ("photovoltaics", "unknown"),
            ("solar radiation", "electricity"),
            year,
        )
        in mappings
    }
    buses.update(
        {
            ((r,), "photovoltaics"): Bus(label=(r, "photovoltaics"))
            for r in pv_regions
        }
    )
    sinks = [
        Sink(
            label=((r,), "pv expansion limit"),
            inputs={
                buses[((r,), "photovoltaics")]: Flow(
                    **invest((key, mappings[key]))
                )
            },
        )
        for r in pv_regions
        for key in [
            Key(
                (r,),
                ("photovoltaics", "unknown"),
                ("solar radiation", "electricity"),
                year,
            )
        ]
    ]

    renewables = ("DE", "renewable share")
    buses[renewables] = buses.get(renewables, Bus(label=renewables))
    renewables = buses[renewables]
    found = find(mappings, "renewable share")
    renewables.share = found[0][1]["renewable share"] if found else 0

    es.add(
        *buses.values(),
        *sinks,
        Source(
            label=("DE", "renewable share compensation"),
            outputs={renewables: Flow()},
        ),
    )

    waste = find(mappings, ("waste", "unknown"))
    assert len(waste) == 1
    waste = waste[0]
    assert waste[0].region[0] == "DE"
    assert waste[0].vectors[0] == "waste"
    es.add(
        Sink(
            label=label(waste, "waste"),
            inputs={
                buses[(waste[0].region[0], waste[0].vectors[0])]: Flow(
                    nominal_value=waste[1]["natural domestic limit"],
                    summed_max=1,
                )
            },
        )
    )

    co2 = find(mappings, ("unknown", "co2"))
    assert len(co2) == 1
    co2 = co2[0]
    assert co2[0].region[0] == "DE"
    assert co2[0].vectors[1] == "co2"
    es.add(
        Sink(
            label=label(co2, "CO2"),
            inputs={
                buses[(co2[0].region[0], co2[0].vectors[1])]: Flow(
                    nominal_value=co2[1].get("emission limit"), summed_max=1
                )
            },
        )
    )

    es.add(
        *[
            Sink(
                label=Label(
                    regions=(rv[0],),
                    technology=("ALL", "ALL"),
                    vectors=(rv[1], rv[1]),
                    name="curtailment",
                ),
                inputs={buses[rv]: Flow()},
            )
            for rv in buses
            if rv[1] == "electricity"
        ]
    )

    Source.slack_costs = sum(
        max(p[1][f"{c} costs"] for p in find(mappings, f"{c} costs"))
        for c in ["variable", "fixed", "capital"]
    )
    es.add(
        *[
            Source(
                label=Label(
                    regions=(rv[0],),
                    technology=("ALL", "ALL"),
                    vectors=(rv[1], rv[1]),
                    name="slack",
                ),
                outputs={buses[rv]: Flow(variable_costs=Source.slack_costs)},
            )
            for rv in buses
            if rv[1] == "electricity"
        ]
    )

    es.add(*demands(mappings, buses))
    es.add(*lines(mappings, buses))
    es.add(*trades(mappings, buses))
    es.add(*fixed(mappings, buses))
    es.add(*flexible(mappings, buses))
    es.add(*storages(mappings, buses))

    return es


def export(mappings, meta, results, year):
    path = Path("results")
    path.mkdir(exist_ok=True)

    header = [
        "id",
        "scenario_id",
        "region",
        "input_energy_vector",
        "output_energy_vector",
        "technology",
        "technology_type",
        "parameter_name",
        "timeindex_start",
        "timeindex_stop",
        "timeindex_resolution",
        "series",
        "unit",
        "tags",
        "method",
        "source",
        "comment",
    ]
    defaults = {
        "scenario_id": 1,
        "unit": "MW/h",
        "tags": "",
        "method": '{"value": "timeseries"}',
        "source": "oemof",
        "comment": "",
    }

    objective = meta["objective"]

    transmissions = [
        k
        for k in results
        if k[1] is not None
        and getattr(k[1].label, "name", "") == "energy flow"
    ]
    sources = [k for k in results if type(k[0]) is Source]
    storages = [
        k for k in results if type(k[0]) is Storage or type(k[1]) is Storage
    ]

    flow = namedtuple("flow", ["key", "source", "name"])
    flows = [flow(line, line[1], line[1].label.name) for line in transmissions]
    flows.extend(
        [
            flow(key, key[0], key[1].label.name)
            for key in results
            if key[1] is not None
            and getattr(key[1].label, "name", "") == "losses"
        ]
    )

    flows.extend(
        [
            # Key[1] is not actually the source, but it should work regardless.
            # Key[0] would be the actual source, but it's a `Bus` without a
            # usable label, so using the target to get the necessary parameters
            # should be a viable workaround.
            flow(key, key[1], key[1].label.name)
            for key in results
            if key[1] is not None
            and getattr(key[1].label, "name", "") == "curtailment"
        ]
    )

    flows.extend(
        [
            flow(key, key[0], key[0].label.name)
            for key in results
            if key[0] is not None
            and getattr(key[0].label, "name", "") == "slack"
        ]
    )

    flows.extend(
        [flow(source, source[0], source[0].label.name) for source in sources]
    )
    flows.extend(
        [
            flow(
                storage,
                *(
                    lambda s: (
                        (s[0], "storage level")
                        if s[1] is None
                        else (s[1], "input energy")
                        if type(s[1]) is Storage
                        else (s[0], "output energy")
                    )
                )(storage),
            )
            for storage in storages
        ]
    )

    series = [
        {
            "region": list(flow.source.label.regions),
            "input_energy_vector": flow.source.label.vectors[0],
            "output_energy_vector": flow.source.label.vectors[1],
            "technology": flow.source.label.technology[0],
            "technology_type": flow.source.label.technology[1],
            "parameter_name": flow.name,
            "timeindex_start": results[flow.key]["sequences"].index[0],
            "timeindex_stop": results[flow.key]["sequences"].index[-1],
            "timeindex_resolution": "1"
            + results[flow.key]["sequences"].index.freq.name.lower(),
            "series": list(results[flow.key]["sequences"].iloc[:, 0]),
        }
        for flow in flows
    ]

    with open(path / "oed_timeseries_output.csv", "w") as f:
        writer = DictWriter(f, fieldnames=header, delimiter=";", quotechar="'")
        writer.writeheader()
        writer.writerows(
            {
                "id": i,
                **{k: str(s[k]).replace("'", '"') for k in s},
                **defaults,
            }
            for i, s in enumerate(series)
        )

    for key in [
        "timeindex_start",
        "timeindex_stop",
        "timeindex_resolution",
        "series",
    ]:
        header.remove(key)
    header.insert(8, "value")
    header.insert(3, "year")

    def group(series):
        return (
            series["region"],
            series["input_energy_vector"],
            series["output_energy_vector"],
            series["technology"],
            series["technology_type"],
            series["parameter_name"],
        )

    series = sorted(series, key=group)
    sums = [
        {
            "region": key[0],
            "input_energy_vector": key[1],
            "output_energy_vector": key[2],
            "technology": key[3],
            "technology_type": key[4],
            "parameter_name": key[5],
            "value": sum(chain(*(row["series"] for row in group))) / 1000,
            "unit": "GWh/a",
        }
        for key, group in groupby(series, group)
    ]

    series = sorted(series, key=lambda row: row["region"])
    emissions = [
        {
            "region": key,
            "input_energy_vector": "ALL",
            "output_energy_vector": "CO2",
            "technology": "ALL",
            "technology_type": "ALL",
            "parameter_name": "emissions",
            "value": sum(
                sum(row["series"]) * m["emission factor"] / m["output ratio"]
                for row in group
                for m in [mappings[Key.from_dictionary(row)]]
            )
            / pow(10, 9),
            "unit": "Gt/a",
        }
        for key, group in groupby(
            (
                row
                for row in series
                if "emission factor"
                in mappings.get(Key.from_dictionary(row), {})
            ),
            lambda row: row["region"],
        )
    ]

    investments = [
        {
            "region": list(label.regions),
            "input_energy_vector": label.vectors[0],
            "output_energy_vector": label.vectors[1],
            "technology": label.technology[0],
            "technology_type": label.technology[1],
            "parameter_name": "added capacity",
            "value": results[key]["scalars"]["invest"].sum() / 1000,
            "unit": "GWh/a" if type(key[0]) is Storage else "GW/a",
        }
        for key in results
        if (type(key[1]) is not Storage)
        and (
            (type(key[0]) is not Storage)
            or ((type(key[0]) is Storage) and (key[1] is None))
        )
        if "invest" in results[key]["scalars"]
        if (
            key[1] is not None
            and tuple(key[1].label)[-1] != "pv expansion limit"
        )
        for label in [key[0].label]
    ]

    total_capacity = [
        {
            "region": list(label.regions),
            "input_energy_vector": label.vectors[0],
            "output_energy_vector": label.vectors[1],
            "technology": label.technology[0],
            "technology_type": label.technology[1],
            "parameter_name": "capacity",
            "value": value / 1000,
            "unit": "GWh/a" if type(key[0]) is Storage else "GW/a",
        }
        for key in results
        for label in [key[0].label]
        if type(label) is Label
        and label.name in ["flow_bus", "electricity generation", "storage"]
        and (type(key[0]) is not Storage or key[1] is None)
        for flow in [key[0].outputs[key[1]] if key[1] is not None else key[0]]
        for value in [
            flow.investment.existing + results[key]["scalars"].invest
            if "invest" in results[key]["scalars"]
            else flow.nominal_storage_capacity
            if type(key[0]) is Storage
            else flow.nominal_value
        ]
    ]

    regions = sorted({region for row in series for region in row["region"]})

    emissions.append(
        reduce(
            lambda d1, d2: {
                **d1,
                "region": sorted(set(d1["region"] + d2["region"])),
                "value": d1["value"] + d2["value"],
            },
            emissions,
        )
    )

    cost_defaults = {
        "region": regions,
        "input_energy_vector": "ALL",
        "output_energy_vector": "ALL",
        "technology": "ALL",
        "technology_type": "ALL",
        "unit": "€/a",
    }
    costs = [
        {
            **cost_defaults,
            "parameter_name": "variable cost",
            "value": objective
            - sum(
                megawatthours * k[0].slack_costs
                for k in results
                if type(k[0]) is Source
                if "slack" == k[0].label.name
                for megawatthours in list(results[k]["sequences"].iloc[:, 0])
            ),
        }
    ]

    costs.append(
        {
            **cost_defaults,
            "parameter_name": "fixed cost",
            "value": sum(
                m["fixed costs"] * m.get("installed capacity", 0)
                for row in series
                for m in [mappings.get(Key.from_dictionary(row), {})]
                if "fixed costs" in m
            ),
        }
    )

    costs.append(
        {
            **cost_defaults,
            "parameter_name": "investment cost",
            "value": sum(
                (
                    key[0].outputs[key[1]] if key[1] is not None else key[0]
                ).investment.ep_costs
                * results[key]["scalars"]["invest"].sum()
                for key in results
                if "invest" in results[key]["scalars"]
                if (
                    key[1] is not None
                    and tuple(key[1].label)[-1] != "pv expansion limit"
                )
            ),
        }
    )

    costs.append(
        {
            **cost_defaults,
            "parameter_name": "system cost",
            "value": sum(cost["value"] for cost in costs),
            "unit": "€",
        }
    )

    def group(row):
        return (
            row["input_energy_vector"],
            row["output_energy_vector"],
            row["technology"],
            row["technology_type"],
            row["parameter_name"],
        )

    series = sorted(series, key=group)
    storage_losses = [
        {
            "region": regions,
            "input_energy_vector": key[0],
            "output_energy_vector": key[1],
            "technology": key[2],
            "technology_type": key[3],
            "parameter_name": "losses",
            "value": sum(
                sum(row["series"])
                * (
                    (1 - m["input ratio"])
                    if row["parameter_name"] == "input energy"
                    else (1 / m["output ratio"] - 1)
                )
                for row in group
                for m in [mappings[Key.from_dictionary(row)]]
            )
            / 1000,
            "unit": "GWh/a",
        }
        for key, group in groupby(
            (
                row
                for row in series
                if row["parameter_name"] in ["input energy", "output energy"]
            ),
            key=group,
        )
    ]

    renewables = {
        "region": regions,
        "input_energy_vector": "ALL RENEW",
        "output_energy_vector": "ALL RENEW",
        "technology": "ALL RENEW",
        "technology_type": "ALL RENEW",
        "parameter_name": "renewable generation",
        "value": sum(
            row["value"]
            for row in sums
            if row["technology"]
            in ["wind turbine", "photovoltaics", "hydro turbine"]
        ),
        "unit": "GWh/a",
    }

    del defaults["unit"]
    defaults["method"] = '{"value": "aggregated"}'
    defaults["year"] = year
    with open(path / "oed_scalar_output.csv", "w") as f:
        writer = DictWriter(f, fieldnames=header, delimiter=";", quotechar="'")
        writer.writeheader()
        writer.writerows(
            {
                "id": i,
                **{k: str(s[k]).replace("'", '"') for k in s},
                **defaults,
            }
            for i, s in enumerate(
                chain(
                    sums,
                    storage_losses,
                    emissions,
                    costs,
                    [renewables],
                    investments,
                    total_capacity,
                )
            )
        )

    header = ["id", "scenario", "region", "year", "source", "comment"]
    row = {
        "id": 1,
        "scenario": "base",
        "region": str({"DE": DE}).replace("'", '"'),
        "year": year,
        "source": "oemof",
        "comment": [
            "The scenario depicts the electricity sector in Germany."
            " It is divided into 18 nodes, 16 nodes as federal states and 2"
            " offshore nodes. Germany's neighbouring countries are not"
            " considered."
        ],
    }
    with open(path / "oed_scenario_output.csv", "w") as f:
        writer = DictWriter(f, fieldnames=header, delimiter=";", quotechar="|")
        writer.writeheader()
        writer.writerow(row)

    package = Package()
    resources = [
        Resource(
            {
                "path": str(path / csvfile),
                "dialect": {"delimiter": ";", "quoteChar": "|"},
                "profile": "tabular-data-resource",
            },
            detector=Detector(field_float_numbers=True),
        )
        for csvfile in [
            "oed_scalar_output.csv",
            "oed_scenario_output.csv",
            "oed_timeseries_output.csv",
        ]
    ]
    for resource in resources:
        resource.infer()
        package.add_resource(resource)
    for r in package.resources:
        r["schema"]["primaryKey"] = ["id"]
        if r["name"] in ["oed_scalar_output", "oed_timeseries_output"]:
            r["schema"]["foreignKeys"] = [
                {
                    "fields": ["scenario_id"],
                    "reference": {
                        "resource": "oed_scenario_output",
                        "fields": ["id"],
                    },
                }
            ]
    package.to_zip(f"oemof{year}.zip")

    return None


def main(silent=True):
    if not silent:
        logger.enable(__name__)
    year = 2016
    mappings = from_json()[year]
    es = build(mappings, year)
    om = Model(es)
    om.solve(solver="cbc")  # , solve_kwargs={'tee': True})
    results = processing.results(om)
    meta = processing.meta_results(om)
    export(mappings, meta, results, 2016)
    return (es, om)


def find(d, *xs, **kws):
    results = [
        (k, d[k])
        for k in d
        if all([(p in chain(k.items(), d[k].items())) for p in kws.items()])
        if all([(x in chain(k, d[k], k.values(), d[k].values())) for x in xs])
    ]
    return results


if __name__ == "__main__":
    main()

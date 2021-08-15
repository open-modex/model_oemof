from collections import namedtuple
from csv import DictWriter, field_size_limit as csv_field_size_limit
from contextlib import contextmanager
from dataclasses import asdict, astuple, dataclass, fields, replace
from datetime import datetime
from functools import reduce
from itertools import chain, groupby
from pathlib import Path
from pprint import pformat as pf
from tempfile import TemporaryDirectory as TD
from typing import Tuple
import json
import re
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
import click
import pandas as pd
import plotly.graph_objects as plt


csv_field_size_limit(sys.maxsize)

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


def rs2df(rs):
    """Convert time dependent optimization results to a dataframe.

    The resulting dataframe has rows indexed by the timeindex and columns
    which are a two level MultiIndex, where the label of the source node is
    the first level and the label of the target node is the second level.
    """

    def label(o):
        return getattr(o, "label", str(o))

    d = {
        (label(k[0]), label(k[1]), name): rs[k]["sequences"][name]
        for k in rs
        for name in rs[k]["sequences"]
    }
    return (
        pd.DataFrame.from_dict(d)
        .sort_index(axis="columns")
        .rename_axis(columns=["source", "target", "values"])
    )


def sankey(df):
    idx = pd.IndexSlice
    sums = df.sum()
    if len(sums.index.levshape) > 2:
        sums = sums.droplevel(list(range(2, len(sums.index.levshape))))
    sums = sums.drop("None", level=1)

    deletable = [
        k
        for k in sums.index
        if len(sums.loc[idx[k[0], :]]) == 1
        if len(sums.get(idx[:, k[0]], [])) == 1
        if list(sums.loc[idx[k[0], :]]) == list(sums.loc[idx[:, k[0]]])
    ]
    for key in deletable:
        sums = sums.set_axis(
            sums.index.map(lambda k: (k[0], key[1]) if k[1] == key[0] else k)
        )
        sums = sums.drop(key)

    compressable = [
        k
        for k in sums.index
        if len(sums.loc[(k[0],)]) == 1
        if len(sums.loc[idx[:, k[1]]]) == 1
    ]
    for key in compressable:
        if sums.get(idx[:, key[0]]) is None:
            # keep = 0
            def keep(k):
                if key[1] == k[0]:
                    return (key[0], k[1])
                return k

        elif sums.get(idx[key[1],]) is None:
            # keep = 1
            def keep(k):
                if k[1] == key[0]:
                    return (k[0], key[1])
                return k

        else:
            raise ValueError(
                f"Can't decide whether to keep source or target of {key}."
            )
        sums = sums.drop(key)
        sums = sums.set_axis(sums.index.map(keep))

    # Transform 'energy flow, XY -> AB: transmission, hvac / electricity,
    # electricity' nodes into flows from "('XY', 'electricity')" to "('AB',
    # 'electricity')" with a label of "transmission, hvac".
    # Remember to redirect the losses and "energy flow (both directions)"
    # flows, though.
    sums = sums.drop([k for k in sums.index if "(both directions)" in k[1]])
    for key in [k for k in sums.index if "energy flow" in k[1]]:
        sums = sums.drop(key)
        sums = sums.set_axis(
            sums.index.map(lambda k: (key[0], k[1]) if k[0] == key[1] else k)
        )

    def parse(label):
        if ": " in label:
            detail, regions, t1, t2, v1, v2 = re.split(", | / |: ", label)
            key = Key(
                region=tuple(regions.split(" -> ")),
                technology=(t1, t2),
                vectors=(v1, v2),
                year=-1,
            )
            key["detail"] = detail
        else:  # should be "(region, vector)" or "((region,), vector)"
            region, detail = re.match(
                "\(([^()]*|\([^()]*,\)), '([^()]*)'\)", label
            ).groups()
            region = re.sub("[()',']", "", region)
            key = Key(
                region=(re.sub("[()',']", "", region),),
                technology=None,
                vectors=None,
                year=-1,
            )
            key["detail"] = detail
        return key

    def label(key):
        if ": " in key:
            match = re.search("^([^,]*), ", key)
            return (match[1], key.replace(match[0], ""))
        # if re.search("'([^']* expansion limit)'")
        return (key, None)

    labels = list(set(parse(l)["detail"] for p in sums.index for l in p))
    l2i = {l: i for i, l in enumerate(labels)}
    color = "rgba(44, 160, 44, {})"
    figure = plt.Figure(
        data=[
            plt.Sankey(
                node=dict(
                    pad=15,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=color.format(0.8),
                ),
                link=dict(
                    source=[l2i[parse(k[0])["detail"]] for k in sums.index],
                    target=[l2i[parse(k[1])["detail"]] for k in sums.index],
                    value=list(sums),
                    label=[
                        (l0 + " | " + l1)
                        if l0 != "None"
                        and l1 != "None"
                        and l0 != l1
                        and "DE:" not in l0
                        and "DE:" not in l1
                        else l0
                        if l0 != "None"
                        else l1
                        if l1 != "None"
                        else None
                        for k in sums.index
                        for l0 in [str(label(k[0])[1])]
                        for l1 in [str(label(k[1])[1])]
                    ],
                    color=color.format(0.4),
                ),
            )
        ]
    )
    figure.update_layout(title_text="Energy System Flows", font_size=10)
    return figure


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
    ratios = {}
    if "E2P ratio" in mapping[1]:
        ratio = 1 / mapping[1]["E2P ratio"]
        # TODO: Figure out whether these have to be multiplied with
        #       `mapping[1]["input ratio"]` or `mapping[1]["output ratio"]`
        #       respectively. See also the corresponding TODO at storage
        #       construction.
        ratios["invest_relation_input_capacity"] = ratio
        ratios["invest_relation_output_capacity"] = ratio
    return {
        **ratios,
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
                and dictionary[key]["units"][value["parameter_name"]]
                == value["unit"]
            ), textwrap.indent(
                f'\n\nParameter\n\n  {value["parameter_name"]}\n'
                f"\nalready present under\n\n  {key}\n"
                f'\nOld value: {dictionary[key][value["parameter_name"]]}'
                f'{dictionary[key]["units"][value["parameter_name"]]}'
                f'\nNew value: {value.get("value", "Series ommitted...")}'
                f'{value["unit"]}',
                "  ",
            )
        else:
            dictionary[key] = {"units": {}}
        dictionary[key][value["parameter_name"]] = (
            value["value"] if "value" in value else value["series"]
        )
        dictionary[key]["units"][value["parameter_name"]] = value["unit"]
    return dictionary


def from_json(path):
    logger.info("Reading JSON.")
    base = {"concrete": slurp(path)}
    for mapping in base:
        logger.debug(
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
        logger.debug(f"\n{mapping} time series boundaries:" f"\n{pf(tsbs)}")
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
    years = sorted(
        pd.to_datetime(ts).year for tsb in tsbs["concrete"] for ts in tsb
    )
    result.update(
        {
            year: {k: reduced[k] for k in reduced if k.year == year}
            for year in years
        }
    )
    # assertion: len(o.region) == 2
    # =>  o.technology == ('transmission', 'hvac')
    # and o.vectors    == ('electricity', 'electricity')
    #
    # len(o.region) > 2 => 16 (DE) or 18
    return result


@dataclass(eq=True, frozen=True)
class Label:
    regions: Tuple[str]
    technology: Tuple[str, str]
    vectors: Tuple[str, str]
    name: str

    def __str__(self):
        return (
            f"{self.name}, {' -> '.join(self.regions)}:"
            f" {', '.join(self.technology)} / {', '.join(self.vectors)}"
        )

    def __iter__(self):
        return astuple(self).__iter__()


def label(mapping, name):
    return Label(
        mapping[0].regions, mapping[0].technology, mapping[0].vectors, name
    )


def demands(buses, mappings):
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


def transmission(buses, line, penalties, ratios):
    loss_bus = Bus(label=label(line, "losses"))
    loss = Sink(
        label=label(line, "loss-sink"),
        inputs={loss_bus: Flow(variable_costs=penalties["transmission"])},
    )
    flow_bus = Bus(label=label(line, "flow-bus"))
    flow = Sink(
        label=label(line, "energy flow (both directions)"),
        inputs={flow_bus: Flow(min=0, **invest(line))},
    )

    lines = [
        Transformer(
            label=replace(label(line, "energy flow"), regions=regions,),
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


def lines(buses, mappings, penalties):
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
        for node in transmission(buses, line, penalties, ratios)
    ]


def trades(buses, mappings):
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


def fixed(buses, mappings):
    sources = [
        Source(
            label=label(source, "electricity generation"),
            outputs={
                auxiliary_bus: Flow(
                    fix=source[1]["capacity factor"],
                    **invest(source),
                    variable_costs=source[1].get("variable costs", 0),
                )
            },
        )
        for source in find(mappings, "capacity factor")
        for renewables in [buses[("DE", "renewables")]]
        for auxiliary_bus in [Bus(label=label(source, "auxiliary-bus"))]
        for transformer in [
            Transformer(
                label=label(source, "splitter",),
                inputs={auxiliary_bus: Flow()},
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
            )
        ]
    ]
    return (
        sources
        + [o for source in sources for o in source.outputs]
        + [t for source in sources for o in source.outputs for t in o.outputs]
    )


def flexible(buses, mappings):
    limits = find(mappings, "natural domestic limit")
    limit_buses = {
        (l[0].regions, l[0].vectors[0]): Bus(label=label(l, "limit bus"))
        for l in limits
    }
    limit_sinks = {
        (l[0].regions, l[0].vectors[0]): Sink(
            label=label(l, "limit sink"),
            inputs={
                limit_buses[(l[0].regions, l[0].vectors[0])]: Flow(
                    nominal_value=l[1]["natural domestic limit"]
                    * (pow(10, 9) / 3600),
                    summed_max=1,
                )
            },
        )
        for l in limits
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
                )
            },
        )
        for f in fueled
        for source_bus in [Bus(label=label(f, "auxiliary-bus"))]
        for renewables in [buses[("DE", "renewables")]]
        for transformer in [
            Transformer(
                label=label(f, "auxiliary-transformer"),
                inputs={source_bus: Flow()},
                outputs={
                    buses[(f[0].regions[0], f[0].vectors[1])]: Flow(),
                    buses[("DE", "co2")]: Flow(),
                    **(
                        {renewables: Flow()}
                        if "geothermal" in f[0].technology
                        else {}
                    ),
                    **(
                        {buses[("DE", "waste")]: Flow()}
                        if "waste" in f[0].vectors
                        else {}
                    ),
                    **(
                        {limit_buses[(f[0].regions, f[0].vectors[0])]: Flow()}
                        if (f[0].regions, f[0].vectors[0]) in limit_buses
                        else {}
                    ),
                },
                conversion_factors={
                    buses[("DE", "co2")]: 1
                    * f[1].get("emission factor", 0)
                    / f[1]["output ratio"],
                    **(
                        {buses[("DE", "waste")]: 1 / f[1]["output ratio"]}
                        if "waste" in f[0].vectors
                        else {}
                    ),
                    **(
                        {
                            limit_buses[(f[0].regions, f[0].vectors[0])]: (
                                1 / f[1]["output ratio"]
                            )
                        }
                        if (f[0].regions, f[0].vectors[0]) in limit_buses
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
        + list(limit_buses.values())
        + list(limit_sinks.values())
    )


def storages(buses, mappings, penalties):
    return [
        Storage(
            label=label(storage, "storage"),
            **investment,
            initial_storage_level=0,
            inflow_conversion_factor=storage[1]["input ratio"],
            outflow_conversion_factor=storage[1]["output ratio"],
            inputs={
                buses[(storage[0].regions[0], storage[0].vectors[0])]: Flow(
                    **nv,
                    variable_costs=storage[1].get("variable costs", 0)
                    + penalties["storage"],
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
        # TODO: Figure out whether these nominal values have to be multiplied
        #       with `storage[1]["input ratio"]` or
        #       `storage[1]["output ratio"]` respectively.
        for nv in [
            {"nominal_value": storage[1].get("installed capacity", 0)}
            if "investment" not in investment
            else {}
        ]
    ]


def build(mappings, penalties, timesteps, year):
    logger.info("Building the energy system.")
    es = ES(
        timeindex=pd.date_range(
            f"{year}-01-01T00:00:00", f"{year}-12-31T23:00:00", freq="1h"
        )[0:timesteps]
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

    demand_sinks = demands(buses, mappings)
    total_demand = sum(
        v for sink in demand_sinks for v in list(sink.inputs.values())[0].fix
    )
    renewables = ("DE", "renewables")
    buses[renewables] = buses.get(renewables, Bus(label=renewables))
    renewables = buses[renewables]
    found = find(mappings, "renewable share")
    renewables.share = found[0][1]["renewable share"] if found else 0

    es.add(
        *buses.values(),
        *sinks,
        Sink(
            label=("DE", "renewable share"),
            inputs={
                renewables: Flow(
                    nominal_value=total_demand, summed_min=renewables.share
                )
            },
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

    es.add(*demand_sinks)
    es.add(*lines(buses, mappings, penalties))
    es.add(*trades(buses, mappings))
    es.add(*fixed(buses, mappings))
    es.add(*flexible(buses, mappings))
    es.add(*storages(buses, mappings, penalties))

    renewable_auxiliary_buses = [
        bus
        for transformer in buses[("DE", "renewables")].inputs
        for bus in transformer.inputs
    ]
    for bus in renewable_auxiliary_buses:
        assert tuple(bus.label)[-1] == "auxiliary-bus"
    es.add(
        Sink(
            label=Label(
                regions=("DE",),
                technology=("renewables", "unknown"),
                vectors=("electricity", "electricity"),
                name="curtailment",
            ),
            inputs={bus: Flow() for bus in renewable_auxiliary_buses},
        )
    )

    return es


@contextmanager
def temporary(path):
    path = Path(path)
    yield path


def export(
    export_prefix,
    mappings,
    meta,
    penalties,
    results,
    temporary_directory,
    year,
):
    logger.info("Exporting the results.")
    base = temporary_directory
    subdirectory = Path("results")
    path = base / subdirectory
    path.mkdir(exist_ok=True)

    store = pd.HDFStore(
        f"{export_prefix.format(year=year)}.results.df.h5", "w",
    )
    df = rs2df(results)
    store["results"] = df.set_axis(
        df.columns.map(lambda xs: tuple(f"{x}" for x in xs)), axis="columns"
    )

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
        "unit": "MW",
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
            "value": value,
            "unit": "GWh",
        }
        for key, group in groupby(series, group)
        for value in [sum(chain(*(row["series"] for row in group))) / 1000]
        if value > 0
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
            "value": value,
            "unit": "Gt",
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
        for value in [
            sum(
                sum(row["series"]) * m["emission factor"] / m["output ratio"]
                for row in group
                for m in [mappings[Key.from_dictionary(row)]]
            )
            / pow(10, 9)
        ]
        if value > 0
    ]

    investments = [
        {
            "region": list(label.regions),
            "input_energy_vector": label.vectors[0],
            "output_energy_vector": label.vectors[1],
            "technology": label.technology[0],
            "technology_type": label.technology[1],
            "parameter_name": "added capacity",
            "value": value,
            "unit": "GWh" if type(key[0]) is Storage else "GW",
        }
        for key in results
        if (type(key[1]) is not Storage)
        and (
            (type(key[0]) is not Storage)
            or ((type(key[0]) is Storage) and (key[1] is None))
        )
        if "invest" in results[key]["scalars"]
        if key[1] is None or tuple(key[1].label)[-1] != "pv expansion limit"
        for label in [key[0].label]
        for value in [results[key]["scalars"]["invest"].sum() / 1000]
        if value > 0
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
            "unit": "GWh" if type(key[0]) is Storage else "GW",
        }
        for key in results
        for label in [key[0].label]
        if type(label) is Label
        and label.name in ["flow-bus", "electricity generation", "storage"]
        and (type(key[0]) is not Storage or key[1] is None)
        for flow in [key[0].outputs[key[1]] if key[1] is not None else key[0]]
        for value in [
            flow.investment.existing + results[key]["scalars"].invest
            if "invest" in results[key]["scalars"]
            else flow.nominal_storage_capacity
            if type(key[0]) is Storage
            else flow.nominal_value
        ]
        if value > 0
    ]

    regions = sorted({region for row in series for region in row["region"]})

    if emissions:
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
        "unit": "€",
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
                for megawatthours in results[k]["sequences"].iloc[:, 0]
            )
            - sum(
                megawatthours * penalties["storage"]
                for k in results
                if type(k[1]) is Storage
                for megawatthours in results[k]["sequences"].iloc[:, 0]
            )
            - sum(
                megawatthours * penalties["transmission"]
                for k in results
                if type(k[1]) is Sink
                if type(k[1].label) is Label
                if k[1].label.name == "loss-sink"
                for megawatthours in results[k]["sequences"].iloc[:, 0]
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

    costs[0]["value"] -= costs[-1]["value"]

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
            "value": value / 1000,
            "unit": "GWh",
        }
        for key, group in groupby(
            (
                row
                for row in series
                if row["parameter_name"] in ["input energy", "output energy"]
            ),
            key=group,
        )
        for value in [
            sum(
                sum(row["series"])
                * (
                    (1 - m["input ratio"])
                    if row["parameter_name"] == "input energy"
                    else (1 / m["output ratio"] - 1)
                )
                for row in group
                for m in [mappings[Key.from_dictionary(row)]]
            )
        ]
        if value > 0
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
        "unit": "GWh",
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
                "path": str(subdirectory / csvfile),
                "dialect": {"delimiter": ";", "quoteChar": "|"},
                "profile": "tabular-data-resource",
            },
            detector=Detector(field_float_numbers=True),
            basepath=str(base),
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
    package.to_zip(f"{export_prefix.format(year=year)}.zip")

    return None


@click.command()
@click.argument(
    "path",
    metavar="<scenario file>",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--export-prefix",
    default="oemof{year}",
    metavar="<prefix>",
    help=(
        "The prefix of files to which result data is exported for from which"
        " it is read. Defaults to `oemof{year}`. Currently, two files are"
        " taken into account:\n\n<prefix>.results.df.h5 - An HDF5 file, where"
        " (sequential) results will be exported to as a `pandas.DataFrame`,"
        " saved under the 'results' key."
        "\n\n<prefix>.zip - A zipped datapackage containing some of the"
        " results in a format conforming to the oedatamodel.\n\nNote, that in"
        " the case of exportation, any existing files will be overwritten."
        " Note also that names appearing enclosed in curly braces will be"
        " replaced with their value. Currently the following such names are"
        " supported:\n\nyear - the value supplied via the `--year` option."
    ),
)
@click.option(
    "--temporary-directory",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    help=(
        "The directory in which to store temporary files. If not specified,"
        " one will created and cleaned up automatically. If the parameter is"
        " specified, the given directory will be created if it doesn't exist"
        " and it will be used to store temporary files, but it will not be"
        " cleaned up automatically, so you can use this parameter to inspect"
        " the temporary files if you need to. Note that any supplied directory"
        " has to be a relative path pointing to a subdirectory of the working"
        " directory."
    ),
)
@click.option(
    "--year", metavar="<year>", required=True, show_default=True, type=int
)
@click.option(
    "--verbosity",
    default="WARNING",
    show_default=True,
    type=click.Choice(
        [
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
            "SILENT",
        ],
        case_sensitive=False,
    ),
    help="Control the verbosity level.",
)
@click.option(
    "--tee/--no-tee",
    default=False,
    show_default=True,
    help="Print solver output.",
)
@click.option(
    "--timesteps",
    default=None,
    metavar="<n>",
    help="Limit the modelled time index to the first <n> steps.",
    type=int,
)
@click.option(
    "--transmission-penalty",
    default=0,
    metavar="<costs>",
    help=(
        "Discourage transmission usage by putting penalty variable <costs> on"
        " transmission losses."
    ),
    show_default=True,
    type=float,
)
@click.option(
    "--storage-penalty",
    default=0,
    metavar="<costs>",
    help=(
        "Discourage storage usage by putting penalty variable <costs> on"
        " storage inputs."
    ),
    show_default=True,
    type=float,
)
def cli(*xs, **ks):
    """Read <scenario file>, build the corresponding model and solve it.

    The <scenario file> should be a JSON file containing all input data.
    """
    ks["penalties"] = {
        "storage": ks["storage_penalty"],
        "transmission": ks["transmission_penalty"],
    }
    del ks["storage_penalty"]
    del ks["transmission_penalty"]
    return main(*xs, **ks)


def main(
    export_prefix,
    path,
    penalties,
    tee,
    temporary_directory,
    timesteps,
    verbosity,
    year,
):
    if verbosity == "SILENT":
        logger.disable(__name__)
    else:
        logger.remove()
        logger.add(sys.stderr, level=verbosity)

    mappings = from_json(path)[year]
    es = build(mappings, penalties, timesteps, year)

    logger.info("Building the model.")
    om = Model(es)

    logger.info("Starting the solver.")
    om.solve(solver="gurobi", solve_kwargs={"tee": tee})

    logger.info("Processing the results.")
    results = processing.results(om)
    meta = processing.meta_results(om)

    with (
        temporary(temporary_directory) if temporary_directory else TD(dir=".")
    ) as td:
        td = Path(td)
        export(export_prefix, mappings, meta, penalties, results, td, year)
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
    cli()

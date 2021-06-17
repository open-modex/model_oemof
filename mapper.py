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

from frictionless import Detector, Package, Resource
from loguru import logger
from oemof.solph import (
    Bus,
    EnergySystem as ES,
    Flow,
    GenericStorage as Storage,
    Model,
    Sink,
    Source,
    Transformer,
    processing,
)
import pandas as pd


csv_field_size_limit(sys.maxsize)

logger.disable(__name__)

mappings = ["concrete"]  # , "normalized"]
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
                else datetime.fromisoformat(str(d["timeindex_start"])).year
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
    keys = (
        [replace(key, region=(kr,)) for kr in key.region]
        if len(key.region) > 2
        else [key]
    )
    for key in keys:
        if key in dictionary:
            assert value["parameter_name"] not in dictionary[key]
        else:
            dictionary[key] = {}
        dictionary[key][value["parameter_name"]] = (
            value["value"] if "value" in value else value["series"]
        )
    logger.info(value)
    return dictionary


def from_json():
    base = {
        mapping: slurp(f"base-scenario.{mapping}.json") for mapping in mappings
    }
    for mapping in mappings:
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
    for mapping in mappings:
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
        for mapping in mappings
    }
    result["concrete"]["objects"] = reduced
    result[2016] = {k: reduced[k] for k in reduced if k.year == 2016}
    result["techs"] = techs(result[2016])
    result["vectors"] = vectors(result[2016])
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
    }
    # TODO: File an issue because there's no (el, el) input/output ratio for
    #       the "Baltic" and the "North" region.
    ratios["Baltic"] = {"ir": 1.0, "or": 0.97}
    ratios["North"] = {"ir": 1.0, "or": 0.97}
    return [
        Transformer(
            label=label(line, "line")._replace(regions=regions),
            inputs={
                source: Flow(
                    min=0, nominal_value=line[1]["installed capacity"]
                )
            },
            outputs={
                target: Flow(
                    min=0, nominal_value=line[1]["installed capacity"]
                )
            },
            conversion_factors={
                source: ratios[regions[0]]["or"],
                target: ratios[regions[1]]["ir"],
            },
        )
        for line in find(
            mappings, "installed capacity", technology=("transmission", "hvac")
        )
        for regions in [line[0].regions, tuple(reversed(line[0].regions))]
        for source in [buses[(regions[0], line[0].vectors[0])]]
        for target in [buses[(regions[1], line[0].vectors[1])]]
    ]


def trades(mappings, buses):
    return [
        Sink(
            label=label(trade, "trade"),
            inputs={
                buses[(trade[0].regions[0], trade[0].vectors[0])]: Flow(
                    fix=trade[1]["trade volume"], nominal_value=1
                )
            },
        )
        for trade in find(mappings, technology=("transmission", "trade"))
    ]


def fixed(mappings, buses):
    return [
        Source(
            label=label(source, "fixed"),
            outputs={
                buses[(source[0].regions[0], source[0].vectors[1])]: Flow(
                    fix=source[1]["capacity factor"],
                    nominal_value=source[1]["installed capacity"],
                    variable_costs=source[1].get("variable costs", 0),
                )
            },
        )
        for source in find(mappings, "capacity factor")
    ]


def flexible(mappings, buses):
    limits = {
        (l[0].regions, l[0].vectors[0]): (
            l[1]["natural domestic limit"] * (pow(10, 9) / 3600)
        )
        for l in find(mappings, "natural domestic limit")
    }
    fueled = find(mappings, "emission factor", "installed capacity")
    co2c = find(mappings, vectors=("unknown", "co2"))[0][1]["emission costs"]
    return [
        Source(
            label=label(f, "flexible"),
            outputs={
                buses[(f[0].regions[0], f[0].vectors[1])]: Flow(
                    nominal_value=f[1]["installed capacity"],
                    variable_costs=(
                        f[1]["variable costs"]
                        + (1 / f[1]["output ratio"])
                        * (f[1]["emission factor"] * co2c + f[1]["fuel costs"])
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
    ]


def storages(mappings, buses):
    return [
        Storage(
            label=label(storage, "storage"),
            nominal_storage_capacity=(
                storage[1]["installed capacity"] * storage[1]["E2P ratio"]
            ),
            inflow_conversion_factor=storage[1]["input ratio"],
            outflow_conversion_factor=storage[1]["output ratio"],
            inputs={
                buses[(storage[0].regions[0], storage[0].vectors[0])]: Flow(
                    nominal_value=storage[1]["installed capacity"],
                    variable_costs=storage[1]["variable costs"],
                )
            },
            outputs={
                buses[(storage[0].regions[0], storage[0].vectors[1])]: Flow(
                    nominal_value=storage[1]["installed capacity"],
                    variable_costs=0,
                )
            },
        )
        for storage in find(mappings, "installed capacity", "E2P ratio")
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

    es.add(*buses.values())
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
        k for k in results if (k[1] is not None) and (k[1].label[-1] == "line")
    ]
    sources = [k for k in results if type(k[0]) is Source]
    storages = [
        k for k in results if type(k[0]) is Storage or type(k[1]) is Storage
    ]

    flow = namedtuple("flow", ["key", "source", "name"])
    flows = [flow(line, line[1], "energy flow") for line in transmissions]
    flows.extend(
        [
            flow(source, source[0], "electricity generation")
            for source in sources
        ]
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

    losses = [
        {**s, "value": s["value"] * 0.03, "parameter_name": "losses"}
        for s in sums
        if s["parameter_name"] == "energy flow"
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
                if "emission factor" in mappings[Key.from_dictionary(row)]
            ),
            lambda row: row["region"],
        )
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
            "value": objective,
        }
    ]

    costs.append(
        {
            **cost_defaults,
            "parameter_name": "fixed cost",
            "value": sum(
                m["fixed costs"]
                for row in series
                for m in [mappings[Key.from_dictionary(row)]]
                if "fixed costs" in m
            ),
        }
    )

    costs.append(
        {
            **cost_defaults,
            "parameter_name": "system cost",
            "value": sum(cost["value"] for cost in costs),
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
    losses.extend(
        [
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
                        1
                        - m[
                            (
                                "in"
                                if row["parameter_name"] == "input energy"
                                else "out"
                            )
                            + "put ratio"
                        ]
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
                    if row["parameter_name"]
                    in ["input energy", "output energy"]
                ),
                key=group,
            )
        ]
    )

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
                chain(sums, losses, emissions, costs, [renewables])
            )
        )

    header = ["id", "scenario", "region", "year", "source", "comment"]
    row = {
        "id": 1,
        "scenario": "base",
        "region": str({"DE": DE}).replace("'", '"'),
        "year": year,
        "source": "",
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
        if all([(x in chain(k, d[k])) for x in xs])
    ]
    return results


if __name__ == "__main__":
    main()

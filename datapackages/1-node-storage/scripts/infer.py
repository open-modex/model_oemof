"""
Run this script from the root directory of the datapackage to update
or create meta data.
"""
from oemof.tabular.datapackage import building


building.infer_metadata(
    package_name='1-node-storage',
    foreign_keys={
        'bus': ['volatile', 'dispatchable', 'storage', 'load'],
        'profile': ['load', 'volatile'],
        'from_to_bus': ['link'],
        'chp': [],
    },
)

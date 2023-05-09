"""Functions for managing worker state.

In general, one uses these by first calling init_* or set_* to create the
attribute, then calling get_* to retrieve the corresponding value.
"""
from functools import partial

from dask.distributed import get_worker

from env_search.warehouse.module import WarehouseConfig, WarehouseModule


#
# Generic
#


def set_worker_state(key: str, val: object):
    """Sets worker_state[key] = val"""
    worker = get_worker()
    setattr(worker, key, val)


def get_worker_state(key: str) -> object:
    """Retrieves worker_state[key]"""
    worker = get_worker()
    return getattr(worker, key)


#
# Warehouse module
#

WAREHOUSE_MOD_ATTR = "warehouse_module"


def init_warehouse_module(config: WarehouseConfig):
    """Initializes this worker's warehouse module."""
    set_worker_state(WAREHOUSE_MOD_ATTR, WarehouseModule(config))


def get_warehouse_module() -> WarehouseModule:
    """Retrieves this worker's warehouse module."""
    return get_worker_state(WAREHOUSE_MOD_ATTR)
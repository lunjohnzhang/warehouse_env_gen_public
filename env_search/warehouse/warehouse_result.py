"""Class representing the results of an evaluation."""
from dataclasses import dataclass, asdict
from typing import List

import numpy as np


def maybe_mean(arr, indices=None):
    """Calculates mean of arr[indices] if possible.

    indices should be a list. If it is None, the mean of the whole arr is taken.
    """
    indices = (slice(len(arr))
               if arr is not None and indices is None else indices)
    return None if arr is None else np.mean(arr[indices], axis=0)


def maybe_median(arr, indices=None):
    """Same as maybe_mean but with median."""
    indices = (slice(len(arr))
               if arr is not None and indices is None else indices)
    return None if arr is None else np.median(arr[indices], axis=0)


def maybe_std(arr, indices=None):
    """Same as maybe_mean but with std."""
    indices = (slice(len(arr))
               if arr is not None and indices is None else indices)
    return None if arr is None else np.std(arr[indices], axis=0)


@dataclass
class WarehouseMetadata:
    """Metadata obtained by running warehouse envs n_evals times"""

    map_int_raw: np.ndarray = None # Generated raw map. Only shelf and floor
    map_int_unrepaired: np.ndarray = None # Generated layout in integer
                                          # (unrepaired)
    map_int: np.ndarray = None  # Generated layout in integer (repaired)
    map_str: List[List[str]] = None
    objs: np.ndarray = None  # Objectives

    throughput : List[float] = None # throughput of the simulation

    tile_usage: np.ndarray = None # (n_eval, n_row, n_col) 3D np array
    # tile_usage: List[List[float]] = None # (n_eval, n_tiles) 2D array
    tile_usage_mean: float = None
    tile_usage_std: float = None

    num_wait: List[List[float]] = None # (n_eval, n_timestep) 2D array
    num_wait_mean: float = None
    num_wait_std: float = None

    finished_task_len: List[List[float]] = None # (n_eval, n_finished_tasks)
                                                # 2D array
    finished_len_mean: float = None
    finished_len_std: float = None

    n_shelf: int = None
    n_endpoint: int = None

    all_task_len_mean: float = None # Average length of all possible
                                    # tasks in the map
    tasks_finished_timestep: List[np.ndarray] = None

    n_shelf_components: int = None # Number of connected shelf components


@dataclass
class WarehouseResult:  # pylint: disable = too-many-instance-attributes
    """Represents `n` results from an objective function evaluation.

    `n` is typically the number of evals (n_evals).

    Different fields are filled based on the objective function.
    """

    ## Raw data ##

    warehouse_metadata: dict = None

    ## Aggregate data ##

    agg_obj: float = None
    agg_measures: np.ndarray = None  # (behavior_dim,) array

    ## Measures of spread ##

    std_obj: float = None
    std_measure: np.ndarray = None  # (behavior_dim,) array

    ## Other data ##

    failed: bool = False
    log_message: str = None

    @staticmethod
    def from_raw(
        warehouse_metadata: WarehouseMetadata,
        opts: dict = None,
    ):
        """Constructs a WarehouseResult from raw data.

        `opts` is a dict with several configuration options. It may be better as
        a gin parameter, but since WarehouseResult is created on workers, gin
        parameters are unavailable (unless we start loading gin on workers too).
        Options in `opts` are:

            `measure_names`: Names of the measures to return
            `aggregation` (default="mean"): How each piece of data should be
                aggregated into single values. Options are:
                - "mean": Take the mean, e.g. mean measure
                - "median": Take the median, e.g. median measure (element-wise)
        """
        # Handle config options.
        opts = opts or {}
        if "measure_names" not in opts:
            raise ValueError("opts should contain `measure_names`")

        opts.setdefault("aggregation", "mean")

        if opts["aggregation"] == "mean":
            agg_obj = maybe_mean(warehouse_metadata.objs)
        elif opts["aggregation"] == "median":
            agg_obj = maybe_median(warehouse_metadata.objs)
        else:
            raise ValueError(f"Unknown aggregation {opts['aggregation']}")

        agg_measures = WarehouseResult._obtain_measure_values(
            asdict(warehouse_metadata), opts["measure_names"])

        return WarehouseResult(
            warehouse_metadata=asdict(warehouse_metadata),
            agg_obj=agg_obj,
            agg_measures=agg_measures,
            # std_obj=maybe_std(objs, std_indices),
            # std_measure=maybe_std(measures, std_indices),
        )

    @staticmethod
    def _obtain_measure_values(metadata, measure_names):
        measures = []
        for measure_name in measure_names:
            measure_val = metadata[measure_name]
            measures.append(measure_val)

        return np.array(measures)

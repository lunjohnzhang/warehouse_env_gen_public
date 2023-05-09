import time
import random
import logging
import pathlib

import traceback
import numpy as np

from typing import List
from env_search.warehouse.warehouse_result import WarehouseResult
from env_search.utils.worker_state import get_warehouse_module
from env_search.warehouse.module import WarehouseModule

logger = logging.getLogger(__name__)


def repair_warehouse(
    map: np.ndarray,
    parent_map: np.ndarray,
    sim_seed: int,
    repair_seed: int,
    w_mode: bool,
    min_n_shelf: int,
    max_n_shelf: int,
    agentNum: int,
):
    start = time.time()

    logger.info("seeding global randomness")
    np.random.seed(sim_seed // np.int32(4))
    random.seed(sim_seed // np.int32(2))

    logger.info("repair warehouse with seed %d", repair_seed)
    warehouse_module = get_warehouse_module()

    try:
        (
            map_json,
            map_np_unrepaired,
            map_comp_unrepaired,
            map_np_repaired,
        ) = warehouse_module.repair(
            map=map,
            parent_map=parent_map,
            sim_seed=sim_seed,
            repair_seed=repair_seed,
            w_mode=w_mode,
            min_n_shelf=min_n_shelf,
            max_n_shelf=max_n_shelf,
            agentNum=agentNum,
        )
    except TimeoutError as e:
        logger.warning(f"repair failed")
        logger.info(f"The map was {map}")
        (
            map_json,
            map_np_unrepaired,
            map_comp_unrepaired,
            map_np_repaired,
        ) = [None] * 4

    logger.info("repair_warehouse done after %f sec", time.time() - start)

    return map_json, map_np_unrepaired, map_comp_unrepaired, map_np_repaired


def run_warehouse(
    map_json: str,
    eval_logdir: pathlib.Path,
    sim_seed: int,
    agentNum: int,
    map_id: int,
    eval_id: int,
) -> WarehouseResult:
    """
    Repair map and run simulation

    Args:
        map (np.ndarray): input map in integer format
        parent_map (np.ndarray): parent solution of the map. Will be None if
                                    current sol is the initial population.
        eval_logdir (str): log dir of simulation
        n_evals (int): number of evaluations
        sim_seed (int): random seed for simulation. Should be different for
                        each solution
        repair_seed (int): random seed for repairing. Should be the same as
                            master seed
        w_mode (bool): whether to run with w_mode, which replace 'r' with
                        'w' in generated map layouts, where 'w' is a
                        workstation. Under w_mode, robots will start from
                        endpoints and their tasks will alternate between
                        endpoints and workstations.
        n_endpt (int): number of endpoint around each obstacle
        min_n_shelf (int): min number of shelves
        max_n_shelf (int): max number of shelves
        agentNum (int): number of drives
        map_id (int): id of the current map to be evaluated. The id
                      is only unique to each batch, NOT to the all the
                      solutions. The id can make sure that each simulation
                      gets a different log directory.
    """
    """Grabs the warehouse module and evaluates level n_evals times."""
    start = time.time()

    logger.info("seeding global randomness")
    np.random.seed(sim_seed // np.int32(4))
    random.seed(sim_seed // np.int32(2))

    logger.info("run warehouse with seed %d", sim_seed)
    warehouse_module = get_warehouse_module()

    try:
        result = warehouse_module.evaluate(
            map_json=map_json,
            eval_logdir=eval_logdir,
            sim_seed=sim_seed,
            agentNum=agentNum,
            map_id=map_id,
            eval_id=eval_id,
        )
    except TimeoutError as e:
        layout = map_json["layout"]
        logger.warning(f"evaluate failed")
        logger.info(f"The map was {layout}")
        result = {}

    logger.info("run_warehouse done after %f sec", time.time() - start)

    return result


def process_warehouse_eval_result(
    curr_result_json: List[dict],
    n_evals: int,
    map_np_unrepaired,
    map_comp_unrepaired,
    map_np_repaired,
    w_mode: bool,
    max_n_shelf: int,
    map_id: int,
):
    start = time.time()

    warehouse_module = get_warehouse_module()

    results = warehouse_module.process_eval_result(
        curr_result_json=curr_result_json,
        n_evals=n_evals,
        map_np_unrepaired=map_np_unrepaired,
        map_comp_unrepaired=map_comp_unrepaired,
        map_np_repaired=map_np_repaired,
        w_mode=w_mode,
        max_n_shelf=max_n_shelf,
        map_id=map_id,
    )
    logger.info("process_warehouse_eval_result done after %f sec",
                time.time() - start)

    return results
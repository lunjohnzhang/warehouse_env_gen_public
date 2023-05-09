"""WarehouseConfig and WarehouseModule.

Usage:
    # Run as a script to demo the WarehouseModule.
    python env_search/warehouse/module.py
"""

import os
import gin
import copy
import json
import time
import fire
import logging
import pathlib
import warnings
import warehouse_sim  # type: ignore # ignore pylance warning
import numpy as np
import multiprocessing

from typing import List
from dataclasses import dataclass
from itertools import repeat
from typing import Collection, Optional
from queue import Queue
from typing import Collection
from env_search import LOG_DIR
from env_search.utils.logging import setup_logging
from env_search.warehouse.milp_repair import repair_env
from env_search.warehouse.warehouse_result import (WarehouseResult,
                                                   WarehouseMetadata)
from env_search.utils import (kiva_obj_types, KIVA_ROBOT_BLOCK_WIDTH,
                              KIVA_WORKSTATION_BLOCK_WIDTH, MIN_SCORE,
                              KIVA_ROBOT_BLOCK_HEIGHT, kiva_env_number2str,
                              kiva_env_str2number, format_env_str,
                              read_in_kiva_map, flip_tiles)

logger = logging.getLogger(__name__)


@gin.configurable
@dataclass
class WarehouseConfig:
    """
    Config warehouse simulation

    Args:
        measure_names (list[str]): list of names of measures
        aggregation_type (str): aggregation over `n_evals` results
        scenario (str): scenario (SORTING, KIVA, ONLINE, BEE)
        task (str): input task file

        cutoffTime (int): cutoff time (seconds)
        screen (int): screen option (0: none; 1: results; 2:all)
        solver (str): solver (LRA, PBS, WHCA, ECBS)
        id (bool): independence detection
        single_agent_solver (str): single-agent solver (ASTAR, SIPP)
        lazyP (bool): use lazy priority
        simulation_time (int): run simulation
        simulation_window (int): call the planner every simulation_window
                                 timesteps
        travel_time_window (int): consider the traffic jams within the
                                  given window
        planning_window (int): the planner outputs plans with first
                                     planning_window timesteps collision-free
        potential_function (str): potential function (NONE, SOC, IC)
        potential_threshold (int): potential threshold
        rotation (bool): consider rotation
        robust (int): k-robust (for now, only work for PBS)
        CAT (bool): use conflict-avoidance table
        hold_endpoints (bool): Hold endpoints from Ma et al, AAMAS 2017
        dummy_paths (bool): Find dummy paths from Liu et al, AAMAS 2019
        prioritize_start (bool): Prioritize waiting at start locations
        suboptimal_bound (int): Suboptimal bound for ECBS
        log (bool): save the search trees (and the priority trees)
        test (bool): whether under testing mode.
        use_warm_up (bool): if True, will use the warm-up procedure. In
                            particular, for the initial population, the solution
                            returned from hamming distance objective will be
                            used. For mutated solutions, the solution of the
                            parent will be used.
        save_result (bool): Whether to allow C++ save the result of simulation
        save_solver (bool): Whether to allow C++ save the result of solver
        save_heuristics_table (bool): Whether to allow C++ save the result of
                                      heuristics table
        stop_at_traffic_jam (bool): whether stop the simulation at traffic jam
        obj_type (str): type of objective
                        ("throughput", "throughput_plus_n_shelf")
    """
    # Measures.
    measure_names: Collection[str] = gin.REQUIRED

    # Results.
    aggregation_type: str = gin.REQUIRED,

    # Simulation
    scenario: str = gin.REQUIRED,
    task: str = gin.REQUIRED,
    cutoffTime: int = gin.REQUIRED,
    screen: int = gin.REQUIRED,
    solver: str = gin.REQUIRED,
    id: bool = gin.REQUIRED,
    single_agent_solver: str = gin.REQUIRED,
    lazyP: bool = gin.REQUIRED,
    simulation_time: int = gin.REQUIRED,
    simulation_window: int = gin.REQUIRED,
    travel_time_window: int = gin.REQUIRED,
    planning_window: int = gin.REQUIRED,
    potential_function: str = gin.REQUIRED,
    potential_threshold: int = gin.REQUIRED,
    rotation: bool = gin.REQUIRED,
    robust: int = gin.REQUIRED,
    CAT: bool = gin.REQUIRED,
    hold_endpoints: bool = gin.REQUIRED,
    dummy_paths: bool = gin.REQUIRED,
    prioritize_start: bool = gin.REQUIRED,
    suboptimal_bound: int = gin.REQUIRED,
    log: bool = gin.REQUIRED,
    test: bool = gin.REQUIRED,
    use_warm_up: bool = gin.REQUIRED,
    hamming_only: bool = gin.REQUIRED,
    save_result : bool = gin.REQUIRED,
    save_solver : bool = gin.REQUIRED,
    save_heuristics_table : bool = gin.REQUIRED,
    stop_at_traffic_jam : bool = gin.REQUIRED,
    obj_type : str = gin.REQUIRED,


class WarehouseModule:
    def __init__(self, config: WarehouseConfig):
        self.config = config

    def repair(
        self,
        map: np.ndarray,
        parent_map: np.ndarray,
        repair_seed: int,
        sim_seed: int,
        w_mode: bool,
        min_n_shelf: int,
        max_n_shelf: int,
        agentNum: int,
    ):
        map_np_unrepaired = copy.deepcopy(map)

        # Create json string for the map
        if self.config.scenario == "KIVA":
            if self.config.obj_type == "throughput_plus_n_shelf":
                assert max_n_shelf == min_n_shelf
            ADDITION_BLOCK_WIDTH = KIVA_WORKSTATION_BLOCK_WIDTH if w_mode \
                                else KIVA_ROBOT_BLOCK_WIDTH
            ADDITION_BLOCK_HEIGHT = 0 if w_mode else KIVA_ROBOT_BLOCK_HEIGHT
            n_row, n_col = map.shape
            # d = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            # # Keep only max_obs_ratio of obstacles
            # n_max_obs = round(n_row * n_col * max_obs_ratio)
            # n_curr_obs = round(np.sum(map))
            # n_rm_obs = n_curr_obs - n_max_obs
            # if n_rm_obs > 0:
            #     all_obs_idx = np.transpose(
            #         np.nonzero(map == kiva_obj_types.index("@")))
            #     to_change = random.sample(list(all_obs_idx), k=n_rm_obs)
            #     for i, j in to_change:
            #         map[i, j] = kiva_obj_types.index(".")

            # map = put_endpoints(map, n_endpt=n_endpt)

            # Stack left and right additional blocks
            l_block, r_block = get_additional_h_blocks(ADDITION_BLOCK_WIDTH,
                                                       n_row, w_mode)
            assert l_block.shape == r_block.shape == (n_row,
                                                      ADDITION_BLOCK_WIDTH)
            map_comp_unrepaired = np.hstack((l_block, map, r_block))
            n_col_comp = n_col + 2 * ADDITION_BLOCK_WIDTH

            # Stack top and bottom additional blocks (At this point, we assume
            # that left and right blocks are stacked)
            n_row_comp = n_row
            if ADDITION_BLOCK_HEIGHT > 0:
                t_block, b_block = \
                    get_additional_v_blocks(ADDITION_BLOCK_HEIGHT,
                                            n_col_comp, w_mode)
                map_comp_unrepaired = np.vstack(
                    (t_block, map_comp_unrepaired, b_block))
                n_row_comp += 2 * ADDITION_BLOCK_HEIGHT

            # Repair environment here
            format_env = format_env_str(kiva_env_number2str(map_comp_unrepaired))

            logger.info(f"Repairing generated environment:\n{format_env}")

            # Limit n_shelf?
            limit_n_shelf = True
            if self.config.obj_type == "throughput_plus_n_shelf":
                limit_n_shelf = False
            # Warm start schema
            warm_up_sols = None
            if self.config.use_warm_up:
                if parent_map is not None:
                    parent_env_str = format_env_str(
                    kiva_env_number2str(parent_map))
                    logger.info(f"Parent warm up solution:\n{parent_env_str}")
                    warm_up_sols = [parent_map]
                # Get the solution from hamming distance objective
                hamming_repaired_env = repair_env(
                    map_comp_unrepaired,
                    agentNum,
                    add_movement=False,
                    min_n_shelf=min_n_shelf,
                    max_n_shelf=max_n_shelf,
                    seed=repair_seed,
                    w_mode=w_mode,
                    warm_envs_np=warm_up_sols,
                    limit_n_shelf=limit_n_shelf,
                )
                hamming_warm_env_str = format_env_str(
                    kiva_env_number2str(hamming_repaired_env))
                logger.info(
                    f"Hamming warm up solution:\n{hamming_warm_env_str}")

                if parent_map is None:
                    warm_up_sols = [hamming_repaired_env]
                else:
                    warm_up_sols = [hamming_repaired_env, parent_map]

            # If hamming only, we just use hamming_repaired_env as the result
            # env
            if self.config.hamming_only:
                map_np_repaired = hamming_repaired_env
            else:
                map_np_repaired = repair_env(
                    map_comp_unrepaired,
                    agentNum,
                    add_movement=True,
                    warm_envs_np=warm_up_sols,
                    min_n_shelf=min_n_shelf,
                    max_n_shelf=max_n_shelf,
                    seed=repair_seed,
                    w_mode=w_mode,
                    limit_n_shelf=limit_n_shelf,
                )

            # Convert map layout to str format
            map_str_repaired = kiva_env_number2str(map_np_repaired)

            format_env = format_env_str(map_str_repaired)
            logger.info(f"\nRepaired result:\n{format_env}")

            # Create json string to map layout
            map_json = json.dumps({
                "name":
                f"sol-seed={sim_seed}",
                "weight":
                False,
                "n_row":
                n_row_comp,
                "n_col":
                n_col_comp,
                "n_endpoint":
                sum(row.count('e') for row in map_str_repaired),
                "n_agent_loc":
                sum(row.count('r') for row in map_str_repaired),
                "maxtime":
                self.config.simulation_time,
                "layout":
                map_str_repaired,
            })

        else:
            NotImplementedError("Other warehouse types not supported yet.")

        return map_json, map_np_unrepaired, map_comp_unrepaired, map_np_repaired


    def evaluate(
        self,
        map_json: str,
        eval_logdir: pathlib.Path,
        sim_seed: int,
        agentNum: int,
        map_id: int,
        eval_id: int,
    ):
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
        output = str(eval_logdir / f"id_{map_id}-sim_{eval_id}-seed={sim_seed}")

        # We need to construct kwargs manually because some parameters
        # must NOT be passed in in order to use the default values
        # defined on the C++ side.
        # It is very dumb but works.

        kwargs = {
            "map" : map_json,
            "output" : output,
            "scenario" : self.config.scenario,
            "task" : self.config.task,
            "agentNum" : agentNum,
            "cutoffTime" : self.config.cutoffTime,
            "seed" : sim_seed,
            "screen" : self.config.screen,
            "solver" : self.config.solver,
            "id" : self.config.id,
            "single_agent_solver" : self.config.single_agent_solver,
            "lazyP" : self.config.lazyP,
            "simulation_time" : self.config.simulation_time,
            "simulation_window" : self.config.simulation_window,
            "travel_time_window" : self.config.travel_time_window,
            "potential_function" : self.config.potential_function,
            "potential_threshold" : self.config.potential_threshold,
            "rotation" : self.config.rotation,
            "robust" : self.config.robust,
            "CAT" : self.config.CAT,
            "hold_endpoints" : self.config.hold_endpoints,
            "dummy_paths" : self.config.dummy_paths,
            "prioritize_start" : self.config.prioritize_start,
            "suboptimal_bound" : self.config.suboptimal_bound,
            "log" : self.config.log,
            "test" : self.config.test,
            "force_new_logdir": True,
            "save_result": self.config.save_result,
            "save_solver": self.config.save_solver,
            "save_heuristics_table": self.config.save_heuristics_table,
            "stop_at_traffic_jam": self.config.stop_at_traffic_jam,
        }

        # For some of the parameters, we do not want to pass them in here
        # to the use the default value defined on the C++ side.
        planning_window = self.config.planning_window
        if planning_window is not None:
            kwargs["planning_window"] = planning_window

        one_sim_result_jsonstr = warehouse_sim.run(**kwargs)

        result_json = json.loads(one_sim_result_jsonstr)
        return result_json


    def process_eval_result(
        self,
        curr_result_json: List[dict],
        n_evals: int,
        map_np_unrepaired: np.ndarray,
        map_comp_unrepaired: np.ndarray,
        map_np_repaired: np.ndarray,
        w_mode: bool,
        max_n_shelf: int,
        map_id: int,
    ):
        """
        Process the evaluation result

        Args:
            results_json (List[dict]): result json of all simulations of 1 map.

        """
        # Collect the results
        keys = curr_result_json[0].keys()
        collected_results = {key: [] for key in keys}
        for result_json in curr_result_json:
            for key in keys:
                collected_results[key].append(result_json[key])

        # Calculate n_shelf and n_endpoint
        # Note: we use the number of searchable block (aka the portion of
        # the layout in the middle) as the totol number of tiles
        tile_ele, tile_cnt = np.unique(map_np_repaired, return_counts=True)
        tile_cnt_dict = dict(zip(tile_ele, tile_cnt))
        n_shelf = tile_cnt_dict[kiva_obj_types.index("@")]
        n_endpoint = tile_cnt_dict[kiva_obj_types.index("e")]

        # Get average length of all tasks
        all_task_len_mean = collected_results.get("avg_task_len")
        # all_task_len_mean = calc_path_len_mean(map_np_repaired, w_mode)
        all_task_len_mean = all_task_len_mean[0]

        logger.info(
            f"Map ID {map_id}: Average length of all possible tasks: {all_task_len_mean}")

        # Calculate number of connected shelf components
        n_shelf_components = calc_num_shelf_components(map_np_repaired)
        logger.info(
            f"Map ID {map_id}: Number of connected shelf components: {n_shelf_components}")

        # Post process result if necessary
        tile_usage = np.array(collected_results.get("tile_usage"))
        tile_usage = tile_usage.reshape(n_evals, *map_np_repaired.shape)
        tasks_finished_timestep = [np.array(x) for x in collected_results.get("tasks_finished_timestep")]

        # Get objective based on type
        objs = None
        if self.config.obj_type == "throughput":
            objs = np.array(collected_results.get("throughput"))
        elif self.config.obj_type == "throughput_plus_n_shelf":
            objs = np.array(collected_results.get("throughput")) - \
                (max_n_shelf - n_shelf)**2 * 0.5
        else:
            return ValueError(
                f"Object type {self.config.obj_type} not supported")

        # Create WarehouseResult object using the mean of n_eval simulations
        # For tile_usage, num_wait, and finished_task_len, the mean is not taken
        metadata = WarehouseMetadata(
            objs=objs,
            throughput=collected_results.get("throughput"),
            map_int_unrepaired=map_comp_unrepaired,
            map_int=map_np_repaired,
            map_int_raw=map_np_unrepaired,
            map_str=kiva_env_number2str(map_np_repaired),
            n_shelf=n_shelf,
            n_endpoint=n_endpoint,
            tile_usage=tile_usage,
            tile_usage_mean=np.mean(collected_results.get("tile_usage_mean")),
            tile_usage_std=np.mean(collected_results.get("tile_usage_std")),
            num_wait=collected_results.get("num_wait"),
            num_wait_mean=np.mean(collected_results.get("num_wait_mean")),
            num_wait_std=np.mean(collected_results.get("num_wait_std")),
            finished_task_len=collected_results.get("finished_task_len"),
            finished_len_mean=np.mean(
                collected_results.get("finished_len_mean")),
            finished_len_std=np.mean(collected_results.get("finished_len_std")),
            all_task_len_mean=all_task_len_mean,
            tasks_finished_timestep=tasks_finished_timestep,
            n_shelf_components=n_shelf_components,
        )
        result = WarehouseResult.from_raw(
            warehouse_metadata=metadata,
            opts={
                "aggregation": self.config.aggregation_type,
                "measure_names": self.config.measure_names,
            },
        )

        return result



    def actual_qd_score(self, objs):
        """Calculates QD score of the given objectives.

        Scores are normalized to be non-negative by subtracting a constant min
        score.

        Args:
            objs: List of objective values.
        """
        objs = np.array(objs)
        objs -= MIN_SCORE
        if np.any(objs < 0):
            warnings.warn("Some objective values are still negative.")
        return np.sum(objs)


logger = logging.getLogger(__name__)
d = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def has_endpoint_around(env_np, i, j, n_endpt=2):
    endpoint_cnt = 0
    n_row, n_col = env_np.shape
    for dx, dy in d:
        n_i, n_j = i + dx, j + dy
        if 0 <= n_i < n_row and 0 <= n_j < n_col:
            if env_np[n_i, n_j] == kiva_obj_types.index("e"):
                endpoint_cnt += 1
                if endpoint_cnt >= n_endpt:
                    return True
    return False


def put_endpoints(map, n_endpt=2):
    # Use a new order of putting endpoints everytime
    cur_d = copy.deepcopy(d)
    # np.random.shuffle(cur_d)

    # Put endpoint around the obstacles
    n_row, n_col = map.shape
    for i in range(n_row):
        for j in range(n_col):
            if map[i, j] == kiva_obj_types.index("@"):
                for dx, dy in cur_d:
                    n_i, n_j = i + dx, j + dy
                    # if in range and the tile is empty space, add endpoint
                    if 0 <= n_i < n_row and \
                        0 <= n_j < n_col and \
                        map[n_i, n_j] == kiva_obj_types.index(".") and \
                        not has_endpoint_around(map, i, j, n_endpt=n_endpt):
                        map[n_i, n_j] = kiva_obj_types.index("e")
    return map


def reaches_goal(loc, goal_locs):
    """
    A `loc` reaches goal if any loc in `goal_locs` is around `loc`.
    """
    x, y = loc
    around_goal = False
    goals = []
    for goal_loc in goal_locs:
        for dx, dy in d:
            n_x = x + dx
            n_y = y + dy
            if tuple([n_x, n_y]) == goal_loc:
                goals.append(goal_loc)
                around_goal = True
    return around_goal, goals


def BFS_path_len(start_loc, goal_locs, env_np, w_mode):
    """
    Find shortest path from start_loc to all goal_locs
    """
    # Set goal loc as none-blocking tile, otherwise it cannot be reached.
    # env_np[goal_locs] = kiva_obj_types.index(".")
    result_path_len = {}
    n_goals = len(goal_locs)
    q = Queue()
    q.put(start_loc)
    seen = set()
    m, n = env_np.shape
    dist_matrix = np.full((m, n), np.inf)
    dist_matrix[start_loc] = 0
    block_idxs = [
        kiva_obj_types.index("@"),
        # kiva_obj_types.index("e"),
        kiva_obj_types.index("w"),
        kiva_obj_types.index("r"),
    ]
    if not w_mode:
        block_idxs.append(kiva_obj_types.index("e"))
    while not q.empty():
        curr = q.get()
        x, y = curr
        around_goal, goals = reaches_goal(curr, goal_locs)
        if around_goal:
            shortest = dist_matrix[x, y] + 1
            for goal_reached in goals:
                result_path_len[goal_reached] = shortest
                goal_locs.remove(goal_reached)

            # print(f"Found goal {goal_reached}")
            # print(f"Remaining number of goals {len(goal_locs)}")

            # All goals found?
            if len(goal_locs) == 0:
                assert len(result_path_len) == n_goals
                return result_path_len

        seen.add(curr)
        for dx, dy in d:
            n_x = x + dx
            n_y = y + dy
            if n_x < m and n_x >= 0 and \
               n_y < n and n_y >= 0 and \
               env_np[n_x,n_y] not in block_idxs and\
               (n_x, n_y) not in seen:
                q.put((n_x, n_y))
                dist_matrix[n_x, n_y] = dist_matrix[x, y] + 1
    raise ValueError(f"Start loc: {start_loc}. Remaining goal: {goal_locs}")


def calc_path_len_mean(repaired_env, w_mode):
    if w_mode:
        start_locs = np.where(repaired_env == kiva_obj_types.index("w"))
    else:
        start_locs = np.where(repaired_env == kiva_obj_types.index("e"))

    start_locs = np.stack(start_locs, axis=1)
    end_locs = np.where(repaired_env == kiva_obj_types.index("e"))
    end_locs = np.stack(end_locs, axis=1)

    path_length_table = {}
    for start_loc in start_locs:
        path_length_table[tuple(start_loc)] = BFS_path_len(
                tuple(start_loc),
                [tuple(end_loc) for end_loc in end_locs],
                copy.deepcopy(repaired_env),
                w_mode,
            )

    all_path_length = []
    for start_loc in start_locs:
        for end_loc in end_locs:
            if tuple(start_loc) != tuple(end_loc):
                all_path_length.append(
                    path_length_table[tuple(start_loc)][tuple(end_loc)])

    return np.mean(all_path_length)


def BFS_shelf_component(start_loc, env_np, env_visited):
    """
    Find all shelves that are connected to the shelf at start_loc.
    """
    # We must start searching from shelf
    assert env_np[start_loc] == kiva_obj_types.index("@")

    q = Queue()
    q.put(start_loc)
    seen = set()
    m, n = env_np.shape
    block_idxs = [
        kiva_obj_types.index("e"),
        kiva_obj_types.index("w"),
        kiva_obj_types.index("r"),
        kiva_obj_types.index("w"),
        kiva_obj_types.index("."),
    ]
    while not q.empty():
        curr = q.get()
        x, y = curr
        env_visited[x,y] = True
        seen.add(curr)
        for dx, dy in d:
            n_x = x + dx
            n_y = y + dy
            if n_x < m and n_x >= 0 and \
               n_y < n and n_y >= 0 and \
               env_np[n_x,n_y] not in block_idxs and\
               (n_x, n_y) not in seen:
                q.put((n_x, n_y))


def calc_num_shelf_components(repaired_env):
    env_visited = np.zeros(repaired_env.shape, dtype=bool)
    n_row, n_col = repaired_env.shape
    n_shelf_components = 0
    for i in range(n_row):
        for j in range(n_col):
            if repaired_env[i,j] == kiva_obj_types.index("@") and\
                not env_visited[i,j]:
                n_shelf_components += 1
                BFS_shelf_component((i, j), repaired_env, env_visited)
    return n_shelf_components


def get_additional_h_blocks(ADDITION_BLOCK_WIDTH, n_row, w_mode):
    """
    Generate additional blocks to horizontally stack to the map on the left and
    right side
    """

    if w_mode:
        # In 'w' mode, horizontally stack the workstations
        # The workstation locations are fixed as the following:
        # 1. Stack workstations on the border of the generated map,
        #    meaning that there is no columns on the left/right side of the
        #    left/right workstations.
        # 2. The first row and last row has no workstations.
        # 3. For the rest of the rows, starting from the second row, put
        # workstations for every three rows, meaning that there are at least
        # two empty cells between each pair of workstations.
        # 4. The left and right side of workstation blocks are symmetrical
        l_block = []
        r_block = []
        for i in range(n_row):
            curr_l_row = None
            curr_r_row = None
            if i == 0 or i == n_row - 1 or (i - 1) % 3 != 0:
                curr_l_row = [
                    kiva_obj_types.index(".")
                    for _ in range(ADDITION_BLOCK_WIDTH)
                ]
                curr_r_row = copy.deepcopy(curr_l_row)
            elif (i - 1) % 3 == 0:
                curr_l_row = [
                    kiva_obj_types.index("w"),
                    kiva_obj_types.index(".")
                ]
                curr_r_row = [
                    kiva_obj_types.index("."),
                    kiva_obj_types.index("w")
                ]
            l_block.append(curr_l_row)
            r_block.append(curr_r_row)
        r_block = np.array(r_block)
        l_block = np.array(l_block)

    else:
        # In 'r' mode, horizontally stack the robot start locations
        # The robot start locations are fixed as the following:
        # 1. Stack robot location blocks on either sides of the generated map
        # 2. On each side, the length of the block is 4
        # 3. The top and bottom rows and the left and right columns have no
        #    robots
        # 4. Starting from the 2nd row, there are 2 robots in the middle column
        # 5. There are at most 3 sequential rows of robots
        # 6. For every 3 rows, append a row of empty space
        r_block = []
        n_robot_row = 0
        for i in range(n_row):
            curr_row = None
            if i == 0 or i == n_row - 1:
                curr_row = [
                    kiva_obj_types.index(".")
                    for _ in range(ADDITION_BLOCK_WIDTH)
                ]
            elif n_robot_row < 3:
                curr_row = [
                    kiva_obj_types.index("."),
                    kiva_obj_types.index("r"),
                    kiva_obj_types.index("r"),
                    kiva_obj_types.index("."),
                ]
                n_robot_row += 1
            elif n_robot_row >= 3:
                curr_row = [
                    kiva_obj_types.index(".")
                    for _ in range(ADDITION_BLOCK_WIDTH)
                ]
                n_robot_row = 0
            r_block.append(curr_row)

        # Under 'r' mode, left and right blocks are the same
        r_block = np.array(r_block)
        l_block = copy.deepcopy(r_block)

    return l_block, r_block


def get_additional_v_blocks(ADDITION_BLOCK_HEIGHT, n_col_comp, w_mode):
    """
    Generate additional blocks to vertically stack to the map on the top and
    bottom
    """
    # Only applicable for r mode
    assert not w_mode
    # We only want even # of cols to make the map symmetrical
    assert n_col_comp % 2 == 0
    t_block = []
    b_block = []
    # For r mode, we need to append additional on top and bottom of the map
    for i in range(ADDITION_BLOCK_HEIGHT):
        # We add 'r' to everywhere except for:
        # 1) the first and block column of each row
        # 2) the 2 columns in the middle of each row
        # 2) the first row for t_block and last row for b_block
        cont_r_length = (n_col_comp - 4) // 2

        if i == 0 or i == ADDITION_BLOCK_HEIGHT - 1:
            t_block.append([
                kiva_obj_types.index(".") for _ in range(n_col_comp)
            ])
        else:
            t_block.append([
                kiva_obj_types.index("."),
                *[kiva_obj_types.index("r") for _ in range(cont_r_length)],
                kiva_obj_types.index("."),
                kiva_obj_types.index("."),
                *[kiva_obj_types.index("r") for _ in range(cont_r_length)],
                kiva_obj_types.index("."),
            ])

    t_block = np.array(t_block)
    b_block = copy.deepcopy(t_block)
    assert t_block.shape == b_block.shape == (ADDITION_BLOCK_HEIGHT, n_col_comp)
    return t_block, b_block

def get_comp_map(
    map,
    seed,
    w_mode,
    n_endpt,
    lvl_height,
):
    """
    Helper function that repair one map using hamming for EM-ME inner
    loop.
    """
    np.random.seed(seed // np.int32(4))

    # Put endpoints in raw maps and repair using hamming distance obj
    ADDITION_BLOCK_WIDTH = KIVA_WORKSTATION_BLOCK_WIDTH if w_mode \
                            else KIVA_ROBOT_BLOCK_WIDTH
    # map = put_endpoints(map, n_endpt=n_endpt)
    l_block, r_block = get_additional_h_blocks(ADDITION_BLOCK_WIDTH, lvl_height,
                                             w_mode)
    map_comp = np.hstack((l_block, map, r_block))

    # Same as MILP, in the surrogate model, we replace 'w' with 'r' under
    # w_mode to use 'r' internally.
    if w_mode:
        map_comp = flip_tiles(
            map_comp,
            'w',
            'r',
        )
    return map_comp



def single_simulation(seed, agent_num, kwargs, results_dir):
    kwargs["seed"] = int(seed)
    output_dir = os.path.join(results_dir,
        f"sim-agent_num={agent_num}-seed={seed}")
    kwargs["output"] = output_dir
    kwargs["agentNum"] = agent_num

    result_jsonstr = warehouse_sim.run(**kwargs)
    result_json = json.loads(result_jsonstr)

    throughput = result_json["throughput"]

    # if result_json["throughput"] > max_obj:
    #     max_obj = result_json["throughput"]

    # if result_json["throughput"] < min_obj:
    #     min_obj = result_json["throughput"]

    # Write result to logdir
    # Load and then dump to format the json
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, f"result.json"), "w") as f:
        f.write(json.dumps(result_json, indent=4))

    # Write kwargs to logdir
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        f.write(json.dumps(kwargs, indent=4))

    # # Increment number of agents if necessary
    # if mode == "inc_agents":
    #     kwargs["agentNum"] += 1

    return throughput

def test_calc_path_len_mean(map_filepath):
    # Read in map
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)
    repaired_env_str = raw_env_json["layout"]
    repaired_env = kiva_env_str2number(repaired_env_str)

    avg_len = calc_path_len_mean(repaired_env, True)
    print(f"Average length (naive BFS): {avg_len}")


def test_calc_num_shelf_components(map_filepath):
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)
    repaired_env_str = raw_env_json["layout"]
    repaired_env = kiva_env_str2number(repaired_env_str)
    n_shelf_components = calc_num_shelf_components(repaired_env)
    print(f"Number of connected shelf components: {n_shelf_components}")

def main(
    warehouse_config,
    map_filepath,
    agent_num=10,
    seed=0,
    n_evals=10,
    n_sim=2, # Run `inc_agents` `n_sim`` times
    mode="constant",
):
    """
    For testing purposes. Graph a map and run one simulation.

    Args:
        mode: "constant", "inc_agents", or "inc_timesteps".
              "constant" will run `n_eval` simulations with the same
              `agent_num`.
              "increase" will run `n_eval` simulations with an inc_agents
              number of `agent_num`.
    """
    setup_logging(on_worker=False)

    gin.parse_config_file(warehouse_config)

    # Read in map
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)

    # Create log dir
    map_name = raw_env_json["name"]
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = time_str + "_" + map_name
    log_dir = os.path.join(LOG_DIR, base_log_dir)
    results_dir = os.path.join(log_dir, "results")
    os.mkdir(log_dir)
    os.mkdir(results_dir)

    # Write map file to logdir
    with open(os.path.join(log_dir, "map.json"), "w") as f:
        f.write(json.dumps(raw_env_json, indent=4))


    # Construct kwargs
    kwargs = {
        "map" : json.dumps(raw_env_json),
        # "output" : log_dir,
        "scenario" : gin.query_parameter("WarehouseConfig.scenario"),
        "task" : gin.query_parameter("WarehouseConfig.task"),
        "agentNum" : agent_num,
        "cutoffTime" : gin.query_parameter("WarehouseConfig.cutoffTime"),
        # "seed" : seed,
        "screen" : gin.query_parameter("WarehouseConfig.screen"),
        "solver" : gin.query_parameter("WarehouseConfig.solver"),
        "id" : gin.query_parameter("WarehouseConfig.id"),
        "single_agent_solver" : gin.query_parameter(
            "WarehouseConfig.single_agent_solver"),
        "lazyP" : gin.query_parameter("WarehouseConfig.lazyP"),
        "simulation_time" : gin.query_parameter(
            "WarehouseConfig.simulation_time"),
        "simulation_window" : gin.query_parameter(
            "WarehouseConfig.simulation_window"),
        "travel_time_window" : gin.query_parameter(
            "WarehouseConfig.travel_time_window"),
        "potential_function" : gin.query_parameter(
            "WarehouseConfig.potential_function"),
        "potential_threshold" : gin.query_parameter(
            "WarehouseConfig.potential_threshold"),
        "rotation" : gin.query_parameter("WarehouseConfig.rotation"),
        "robust" : gin.query_parameter("WarehouseConfig.robust"),
        "CAT" : gin.query_parameter("WarehouseConfig.CAT"),
        "hold_endpoints" : gin.query_parameter(
            "WarehouseConfig.hold_endpoints"),
        "dummy_paths" : gin.query_parameter("WarehouseConfig.dummy_paths"),
        "prioritize_start" : gin.query_parameter(
            "WarehouseConfig.prioritize_start"),
        "suboptimal_bound" : gin.query_parameter(
            "WarehouseConfig.suboptimal_bound"),
        "log" : gin.query_parameter("WarehouseConfig.log"),
        "test" : gin.query_parameter("WarehouseConfig.test"),
        "force_new_logdir": False,
        "save_result": gin.query_parameter("WarehouseConfig.save_result"),
        "save_solver": gin.query_parameter("WarehouseConfig.save_solver"),
        "save_heuristics_table": gin.query_parameter("WarehouseConfig.save_heuristics_table"),
        "stop_at_traffic_jam": gin.query_parameter("WarehouseConfig.stop_at_traffic_jam")
    }

    # For some of the parameters, we do not want to pass them in here
    # to the use the default value defined on the C++ side.
    try:
        planning_window = gin.query_parameter("WarehouseConfig.planning_window")
        if planning_window is not None:
            kwargs["planning_window"] = planning_window
    except ValueError:
        pass

    n_workers = 32
    pool = multiprocessing.Pool(n_workers)
    if mode == "inc_agents":
        seeds = []
        agent_nums = []
        for i in range(n_sim):
            seeds.extend(repeat(seed + i, n_evals))
            agent_nums.extend([agent_num + j for j in range(n_evals)])

        throughputs = pool.starmap(
            single_simulation,
            zip(seeds,
                agent_nums,
                repeat(kwargs, n_evals * n_sim),
                repeat(results_dir, n_evals * n_sim)),
        )
    elif mode == "constant":
        agent_nums = [agent_num for _ in range(n_evals)]
        seeds = np.random.choice(np.arange(10000), size=n_evals, replace=False)

        throughputs = pool.starmap(
            single_simulation,
            zip(seeds,
                agent_nums,
                repeat(kwargs, n_evals),
                repeat(results_dir, n_evals)),
        )

    avg_obj = np.mean(throughputs)
    max_obj = np.max(throughputs)
    min_obj = np.min(throughputs)

    n_evals = n_evals * n_sim if mode == "inc_agents" else n_evals

    print(f"Average throughput over {n_evals} simulations: {avg_obj}")
    print(f"Max throughput over {n_evals} simulations: {max_obj}")
    print(f"Min throughput over {n_evals} simulations: {min_obj}")



if __name__ == "__main__":
    fire.Fire(main)

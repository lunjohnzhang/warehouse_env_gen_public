"""Provides MazeManager."""
import logging
from pathlib import Path
from typing import List, Tuple

import gin
import numpy as np
import copy
from dask.distributed import Client
from logdir import LogDir

# from env_search.maze.agents.rl_agent import RLAgentConfig
from env_search.warehouse.emulation_model.buffer import Experience
from env_search.warehouse.emulation_model.aug_buffer import AugExperience
from env_search.warehouse.emulation_model.double_aug_buffer import DoubleAugExperience
from env_search.warehouse.emulation_model.emulation_model import WarehouseEmulationModel
from env_search.warehouse.emulation_model.networks import (
    WarehouseAugResnetOccupancy, WarehouseAugResnetRepairedMapAndOccupancy)
from env_search.warehouse.module import (WarehouseModule, WarehouseConfig,
                                         get_comp_map, get_additional_h_blocks,
                                         get_additional_v_blocks)
from env_search.warehouse.run import (run_warehouse, repair_warehouse,
                                      process_warehouse_eval_result)
from env_search.utils.worker_state import init_warehouse_module

from env_search.utils import (kiva_obj_types, KIVA_ROBOT_BLOCK_WIDTH,
                              KIVA_WORKSTATION_BLOCK_WIDTH,
                              KIVA_ROBOT_BLOCK_HEIGHT, MIN_SCORE,
                              kiva_env_number2str, format_env_str,
                              read_in_kiva_map, flip_tiles)

logger = logging.getLogger(__name__)


@gin.configurable(denylist=["client", "rng"])
class WarehouseManager:
    """Manager for the warehouse environments.

    Args:
        client: Dask client for distributed compute.
        rng: Random generator. Can be set later. Uses `np.random.default_rng()`
            by default.
        n_evals: Number of times to evaluate each solution during real
            evaluation.
        lvl_width: Width of the level.
        lvl_height: Height of the level.
        num_objects: Number of objects in the level to generate.
        min_n_shelf (int): min number of shelves
        max_n_shelf (int): max number of shelves
        w_mode (bool): whether to run with w_mode, which replace 'r' with 'w' in
                       generated map layouts, where 'w' is a workstation.
                       Under w_mode, robots will start from endpoints and their
                       tasks will alternate between endpoints and workstations.
        n_endpt (int): number of endpoint around each obstacle
        agent_num (int): number of drives
    """

    def __init__(
        self,
        client: Client,
        logdir: LogDir,
        rng: np.random.Generator = None,
        n_evals: int = gin.REQUIRED,
        lvl_width: int = gin.REQUIRED,
        lvl_height: int = gin.REQUIRED,
        num_objects: int = gin.REQUIRED,
        min_n_shelf: int = gin.REQUIRED,
        max_n_shelf: int = gin.REQUIRED,
        w_mode: bool = gin.REQUIRED,
        n_endpt: bool = gin.REQUIRED,
        agent_num: int = gin.REQUIRED,
    ):
        self.client = client
        self.rng = rng or np.random.default_rng()

        self.n_evals = n_evals
        self.eval_batch_idx = 0  # index of each batch of evaluation

        self.logdir = logdir

        self.lvl_width = lvl_width
        self.lvl_height = lvl_height

        self.num_objects = num_objects

        self.min_n_shelf = min_n_shelf
        self.max_n_shelf = max_n_shelf

        self.w_mode = w_mode
        self.n_endpt = n_endpt
        self.agent_num = agent_num

        # Set up a module locally and on workers. During evaluations,
        # repair_and_run_warehouse retrieves this module and uses it to
        # evaluate the function. Configuration is done with gin (i.e. the
        # params are in the config file).
        self.module = WarehouseModule(config := WarehouseConfig())
        client.register_worker_callbacks(lambda: init_warehouse_module(config))

        self.emulation_model = None

    def em_init(self,
                seed: int,
                pickle_path: Path = None,
                pytorch_path: Path = None):
        """Initialize the emulation model and optionally load from saved state.

        Args:
            seed: Random seed to use.
            pickle_path: Path to the saved emulation model data (optional).
            pytorch_path: Path to the saved emulation model network (optional).
        """
        self.emulation_model = WarehouseEmulationModel(seed=seed + 420)
        if pickle_path is not None:
            self.emulation_model.load(pickle_path, pytorch_path)
        logger.info("Emulation Model: %s", self.emulation_model)

    def get_initial_sols(self, size: Tuple):
        """Returns random solutions with the given size.

        Args:
            size: Tuple with (n_solutions, sol_size).

        Returns:
            Randomly generated solutions.
        """
        batch_size, solution_dim = size
        if self.num_objects == 2:
            # If we know the exact number of shelves and we have one only
            # two objects (floor or shelf), we can generate solutions
            # directly.
            if self.min_n_shelf == self.max_n_shelf:
                n_shelf = self.min_n_shelf
                idx_array = np.tile(np.arange(solution_dim), (batch_size, 1))
                shelf_idxs = self.rng.permuted(idx_array, axis=1)[:, :n_shelf]
                sols = np.zeros((batch_size, solution_dim), dtype=int)
                for i in range(batch_size):
                    sols[i, shelf_idxs[i]] = 1
                assert np.sum(sols) == batch_size * n_shelf
            else:
                # If we still have only 2 objects, we can generate
                # solutions in a biased fashion and keep generate until we
                # have a the specified number of shelves.
                if self.num_objects == 2:
                    sols = []
                    for _ in range(batch_size):
                        # Keep generate new solutions until we get desired
                        # number of shelves
                        sol = np.ones(solution_dim, dtype=int)
                        while not (self.min_n_shelf <= np.sum(sol) <=
                                   self.max_n_shelf):
                            sol = self.rng.choice(
                                np.arange(self.num_objects),
                                size=solution_dim,
                                p=[
                                    1 - self.max_n_shelf / solution_dim,
                                    self.max_n_shelf / solution_dim
                                ],
                            )
                        sols.append(sol)
        # If we have more than 2 objects, we just generate new
        # solutions directly
        else:
            sols = self.rng.integers(self.num_objects,
                                     size=(batch_size, solution_dim))

        return np.array(sols), None

    def em_train(self):
        self.emulation_model.train()

    def emulation_pipeline(self, sols):
        """Pipeline that takes solutions and uses the emulation model to predict
        the objective and measures.

        Args:
            sols: Emitted solutions.

        Returns:
            lvls: Generated levels.
            objs: Predicted objective values.
            measures: Predicted measure values.
            success_mask: Array of size `len(lvls)`. An element in the array is
                False if some part of the prediction pipeline failed for the
                corresponding solution.
        """
        n_maps = len(sols)
        maps = np.array(sols).reshape(
            (n_maps, self.lvl_height, self.lvl_width)).astype(int)

        # Add l and r block in a batched fashion
        ADDITION_BLOCK_WIDTH = KIVA_WORKSTATION_BLOCK_WIDTH if self.w_mode \
                            else KIVA_ROBOT_BLOCK_WIDTH
        ADDITION_BLOCK_HEIGHT = 0 if self.w_mode else KIVA_ROBOT_BLOCK_HEIGHT

        l_block, r_block = get_additional_h_blocks(ADDITION_BLOCK_WIDTH,
                                                   self.lvl_height, self.w_mode)

        # Repeat boths blocks by n_maps times
        l_blocks = np.tile(l_block, reps=(n_maps, 1, 1))
        r_blocks = np.tile(r_block, reps=(n_maps, 1, 1))
        map_comps = np.concatenate((l_blocks, maps, r_blocks), axis=2)

        if ADDITION_BLOCK_HEIGHT > 0:
            n_col_comp = self.lvl_width + 2 * ADDITION_BLOCK_WIDTH
            t_block, b_block = \
                get_additional_v_blocks(ADDITION_BLOCK_HEIGHT,
                                        n_col_comp, self.w_mode)
            t_blocks = np.tile(t_block, reps=(n_maps, 1, 1))
            b_blocks = np.tile(b_block, reps=(n_maps, 1, 1))
            map_comps = np.concatenate((t_blocks, map_comps, b_blocks), axis=1)


        # Same as MILP, in the surrogate model, we replace 'w' with 'r' under
        # w_mode to use 'r' internally.
        if self.w_mode:
            map_comps = flip_tiles(
                map_comps,
                'w',
                'r',
            )

        # futures = [
        #     self.client.submit(
        #         get_comp_map,
        #         map=map,
        #         seed=self.seed, # This is the master seed
        #         w_mode=self.w_mode,
        #         n_endpt=self.n_endpt,
        #         lvl_height=self.lvl_height,
        #     ) for map in maps
        # ]

        # map_comps = self.client.gather(futures)
        # map_comps = np.array(map_comps)

        assert map_comps.shape == (
            n_maps,
            self.lvl_height + 2 * ADDITION_BLOCK_HEIGHT,
            self.lvl_width + 2 * ADDITION_BLOCK_WIDTH,
        )

        success_mask = np.ones(len(map_comps), dtype=bool)
        objs, measures = self.emulation_model.predict(map_comps)
        return map_comps, objs, measures, success_mask

    def eval_pipeline(self, sols, parent_sols=None, batch_idx=None):
        """Pipeline that takes a solution and evaluates it.

        Args:
            sols: Emitted solution.
            parent_sols: Parent solution of sols.

        Returns:
            Results of the evaluation.
        """
        n_lvls = len(sols)
        lvls = np.array(sols).reshape(
            (n_lvls, self.lvl_height, self.lvl_width)).astype(int)

        if parent_sols is not None:
            parent_lvls = np.array(parent_sols)
        else:
            parent_lvls = [None] * n_lvls

        # Make each solution evaluation have a different seed. Note that we
        # assign seeds to solutions rather than workers, which means that we
        # are agnostic to worker configuration.
        evaluation_seeds = self.rng.integers(np.iinfo(np.int32).max / 2,
                                             size=len(sols),
                                             endpoint=True)

        # Split repair and evaluate.
        # Since evaluation might take a lot longer than repair, and each
        # evaluation might includes several simulations, we want to distribute
        # all the simulations to the workers instead of evaluations to fully
        # exploit the available compute

        # First, repair the maps
        repair_futures = [
            self.client.submit(
                repair_warehouse,
                map=lvl,
                parent_map=parent_lvl,
                sim_seed=seed,
                repair_seed=self.seed,
                w_mode=self.w_mode,
                min_n_shelf=self.min_n_shelf,
                max_n_shelf=self.max_n_shelf,
                agentNum=self.agent_num,
            ) for lvl, parent_lvl, seed in zip(lvls, parent_lvls,
                                               evaluation_seeds)
        ]

        repair_results = self.client.gather(repair_futures)

        # Based on number of simulations (n_evals), create maps and
        # corresponding variables to simulate
        map_jsons_sim = []
        map_np_unrepaired_sim = []
        map_comp_unrepaired_sim = []
        map_np_repaired_sim = []
        maps_id_sim = []
        maps_eval_seed_sim = []
        eval_id_sim = []
        map_ids = np.arange(len(sols))
        for map_id, repair_result, eval_seed in zip(
                map_ids,
                repair_results,
                evaluation_seeds,
        ):
            (
                map_json,
                map_np_unrepaired,
                map_comp_unrepaired,
                map_np_repaired,
            ) = repair_result
            for j in range(self.n_evals):
                map_jsons_sim.append(copy.deepcopy(map_json))
                map_np_unrepaired_sim.append(copy.deepcopy(map_np_unrepaired))
                map_comp_unrepaired_sim.append(
                    copy.deepcopy(map_comp_unrepaired))
                map_np_repaired_sim.append(copy.deepcopy(map_np_repaired))
                maps_id_sim.append(map_id)
                maps_eval_seed_sim.append(eval_seed + j)
                eval_id_sim.append(j)

        # Then, evaluate the maps
        if batch_idx is None:
            batch_idx = self.eval_batch_idx
        eval_logdir = self.logdir.pdir(f"evaluations/eval_batch_{batch_idx}")
        self.eval_batch_idx += 1
        sim_futures = [
            self.client.submit(
                run_warehouse,
                map_json=map_json,
                eval_logdir=eval_logdir,
                sim_seed=sim_seed,
                agentNum=self.agent_num,
                map_id=map_id,
                eval_id=eval_id,
            ) for (
                map_json,
                sim_seed,
                map_id,
                eval_id,
            ) in zip(
                map_jsons_sim,
                maps_eval_seed_sim,
                maps_id_sim,
                eval_id_sim,
            )
        ]
        logger.info("Collecting evaluations")
        results_json = self.client.gather(sim_futures)

        results_json_sorted = []
        for i in range(len(sols)):
            curr_eval_results = []
            for j in range(self.n_evals):
                curr_eval_results.append(results_json[i * self.n_evals + j])
            results_json_sorted.append(curr_eval_results)

        logger.info("Processing eval results")

        process_futures = [
            self.client.submit(
                process_warehouse_eval_result,
                curr_result_json=curr_result_json,
                n_evals=self.n_evals,
                map_np_unrepaired=map_np_unrepaired_sim[map_id * self.n_evals],
                map_comp_unrepaired=map_comp_unrepaired_sim[map_id *
                                                            self.n_evals],
                map_np_repaired=map_np_repaired_sim[map_id * self.n_evals],
                w_mode=self.w_mode,
                max_n_shelf=self.max_n_shelf,
                map_id=map_id,
            ) for (
                curr_result_json,
                map_id,
            ) in zip(
                results_json_sorted,
                map_ids,
            )
        ]
        results = self.client.gather(process_futures)
        return results

    def add_experience(self, sol, result):
        """Add required experience to the emulation model based on the solution
        and the results.

        Args:
            sol: Emitted solution.
            result: Evaluation result.
        """
        obj = result.agg_obj
        meas = result.agg_measures
        input_lvl = result.warehouse_metadata["map_int_unrepaired"]
        repaired_lvl = result.warehouse_metadata["map_int"]

        # Same as MILP, we replace 'w' with 'r' and use 'r' internally in
        # emulation model
        if self.w_mode:
            input_lvl = flip_tiles(input_lvl, 'w', 'r')
            repaired_lvl = flip_tiles(repaired_lvl, 'w', 'r')

        if self.emulation_model.pre_network is not None:
            # Mean of tile usage over n_evals
            avg_tile_usage = np.mean(result.warehouse_metadata["tile_usage"],
                                     axis=0)
            if isinstance(self.emulation_model.pre_network,
                          WarehouseAugResnetOccupancy):
                self.emulation_model.add(
                    AugExperience(sol, input_lvl, obj, meas, avg_tile_usage))
            elif isinstance(self.emulation_model.pre_network,
                            WarehouseAugResnetRepairedMapAndOccupancy):
                self.emulation_model.add(
                    DoubleAugExperience(sol, input_lvl, obj, meas,
                                        avg_tile_usage, repaired_lvl))
        else:
            self.emulation_model.add(Experience(sol, input_lvl, obj, meas))

    @staticmethod
    def add_failed_info(sol, result) -> dict:
        """Returns a dict containing relevant information about failed levels.

        Args:
            sol: Emitted solution.
            result: Evaluation result.

        Returns:
            Dict with failed level information.
        """
        failed_level_info = {
            "solution": sol,
            "level": result.maze_metadata["level"],
            "log_message": result.log_message,
        }
        return failed_level_info

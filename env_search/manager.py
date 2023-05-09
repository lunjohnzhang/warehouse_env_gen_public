"""Provides a class for running each QD algorithm."""
import dataclasses
import itertools
import logging
import pickle as pkl
from typing import Callable, List, Tuple

import cloudpickle
import gin
import numpy as np
from dask.distributed import Client
from logdir import LogDir
from ribs.archives import ArchiveBase, Elite

from env_search.archives import GridArchive
from env_search.emitters import ImprovementEmitter, MapElitesBaselineEmitter, RandomEmitter
from env_search.optimizers import Optimizer
from env_search.utils.logging import worker_log
from env_search.utils.metric_logger import MetricLogger
from env_search.warehouse.warehouse_manager import WarehouseManager

# Just to get rid of pylint warning about unused import (adding a comment after
# each line above messes with formatting).
IMPORTS_FOR_GIN = (
    GridArchive,
    WarehouseManager,
)

EMITTERS_WITH_RESTARTS = (
    ImprovementEmitter,
    MapElitesBaselineEmitter,
    RandomEmitter,
)

logger = logging.getLogger(__name__)


@gin.configurable
class Manager:  # pylint: disable = too-many-instance-attributes
    """Runs an (emulation model) QD algorithm on distributed compute.

    If you are trying to understand this code, first refer to how the general
    pyribs loop works (https://pyribs.org). Essentially, the execute() method of
    this class runs this loop but in a more complicated fashion, as we want to
    distribute the solution evaluations, log various performance metrics, save
    various pieces of data, support reloading / checkpoints, etc.

    Main args:
        client: Dask client for distributed compute.
        logdir: Directory for saving all logging info.
        seed: Master seed. The seed is not passed in via gin because it needs to
            be flexible.
        reload: If True, reload the experiment from the given logging directory.
        env_manager_class: This class calls a separate manager based on the
            environment, such as MazeManager. Pass this class using this
            argument.

    Algorithm args:
        is_em: Whether this algorithm uses emulation models (EM).
        max_evals: Total number of evaluations of the true objective.
        initial_sols: Number of initial solutions to evaluate.
        inner_itrs: Number of times to run the inner loop.
        archive_type: Archive class for both main and emulation archives.
            Intended for gin configuration.
        sol_size: Size of the solution that the emitter should emit and the
            archive should store.
        emitter_types: List of tuples of (class, n); where each tuple indicates
            there should be n emitters with the given class. If is_em, these
            emitters are only used in the inner loop; otherwise, they are
            maintained for the entire run. Intended for gin configuration.
        num_elites_to_eval: Number of elites in the emulation archive to
            evaluate. Pass None to evaluate all elites. (default: None)
        random_sample_em: True if num_elites_to_eval should be selected
            randomly. If num_elites_to_eval is None, this argument is
            ignored. (default: False)
        downsample_em: Whether to downsample the emulation archive.
        downsample_archive_type: Archive type for downsampling. Used for Gin.

    Logging args:
        archive_save_freq: Number of outer itrs to wait before saving the full
            archive (i.e. including solutions and metadata). Set to None to
            never save (the archive will still be available in the reload file).
            Set to -1 to only save on the final iter.
        save_surrogate_archive: Whether to save surrogate archive or not.
        reload_save_freq: Number of outer itrs to wait before saving
            reload data.
        plot_metrics_freq: Number of outer itrs to wait before displaying text
            plot of metrics. Plotting is not expensive, but the output can be
            pretty large.
    """
    def __init__(
        self,
        ## Main args ##
        client: Client,
        logdir: LogDir,
        seed: int,
        reload: bool = False,
        env_manager_class: Callable = gin.REQUIRED,
        ## Algorithm args ##
        is_em: bool = gin.REQUIRED,
        max_evals: int = gin.REQUIRED,
        initial_sols: int = gin.REQUIRED,
        inner_itrs: int = gin.REQUIRED,
        archive_type: Callable = gin.REQUIRED,
        sol_size: int = gin.REQUIRED,
        emitter_types: List[Tuple] = gin.REQUIRED,
        num_elites_to_eval: int = None,
        random_sample_em: bool = False,
        downsample_em: bool = False,
        downsample_archive_type: Callable = None,
        ## Logging args ##
        archive_save_freq: int = None,
        save_surrogate_archive: bool = True,
        reload_save_freq: int = 5,
        plot_metrics_freq: int = 5,
    ):  # pylint: disable = too-many-arguments, too-many-branches

        # Main.
        self.client = client
        self.logdir = logdir

        # Algorithm.
        self.is_em = is_em
        self.max_evals = max_evals
        self.inner_itrs = inner_itrs
        self.initial_sols = initial_sols
        self.archive_type = archive_type
        self.sol_size = sol_size
        self.emitter_types = emitter_types
        self.num_elites_to_eval = num_elites_to_eval
        self.random_sample_em = random_sample_em
        self.downsample_em = downsample_em
        self.downsample_archive_type = downsample_archive_type

        # Logging.
        self.archive_save_freq = archive_save_freq
        self.save_surrogate_archive = save_surrogate_archive
        self.reload_save_freq = reload_save_freq
        self.plot_metrics_freq = plot_metrics_freq

        # Set up the environment manager.
        self.env_manager = env_manager_class(self.client, self.logdir)

        # The attributes below are either reloaded or created fresh. Attributes
        # added below must be added to the _save_reload_data() method.
        if not reload:
            logger.info("Setting up fresh components")
            self.rng = np.random.default_rng(seed)
            self.outer_itrs_completed = 0
            self.evals_used = 0

            metric_list = [
                ("Total Evals", True),
                ("Mean Evaluation", False),
                ("Actual QD Score", True),
                ("Archive Size", True),
                ("Archive Coverage", True),
                ("Best Objective", False),
                ("Worst Objective", False),
                ("Mean Objective", False),
                ("Overall Min Objective", False),
            ]

            self.metrics = MetricLogger(metric_list)
            self.total_evals = 0
            self.overall_min_obj = np.inf

            self.metadata_id = 0
            self.cur_best_id = None  # ID of most recent best solution.

            self.failed_levels = []

            if self.is_em:
                logger.info("Setting up emulation model and archive")
                # Archive must be initialized since there is no optimizer.
                self.env_manager.em_init(seed)
                self.archive: ArchiveBase = archive_type(seed=seed,
                                                         dtype=np.float32)
                self.archive.initialize(self.sol_size)
                logger.info("Archive: %s", self.archive)
            else:
                logger.info("Setting up optimizer for classic pyribs")
                _, self.optimizer = self.build_emitters_and_optimizer(
                    archive_type(seed=seed, dtype=np.float32))
                logger.info("Optimizer: %s", self.optimizer)
                # Set self.archive too for ease of reference.
                self.archive = self.optimizer.archive
                logger.info("Archive: %s", self.archive)
        else:
            logger.info("Reloading optimizer and other data from logdir")

            with open(self.logdir.pfile("reload.pkl"), "rb") as file:
                data = pkl.load(file)
                self.rng = data["rng"]
                self.outer_itrs_completed = data["outer_itrs_completed"]
                self.total_evals = data["total_evals"]
                self.metrics = data["metrics"]
                self.overall_min_obj = data["overall_min_obj"]
                self.metadata_id = data["metadata_id"]
                self.cur_best_id = data["cur_best_id"]
                self.failed_levels = data["failed_levels"]
                if self.is_em:
                    self.archive = data["archive"]
                else:
                    self.optimizer = data["optimizer"]
                    self.archive = self.optimizer.archive

            if self.is_em:
                self.env_manager.em_init(seed,
                                         self.logdir.pfile("reload_em.pkl"),
                                         self.logdir.pfile("reload_em.pth"))

            logger.info("Outer itrs already completed: %d",
                        self.outer_itrs_completed)
            logger.info("Execution continues from outer itr %d (1-based)",
                        self.outer_itrs_completed + 1)
            logger.info("Reloaded archive: %s", self.archive)

        logger.info("solution_dim: %d", self.archive.solution_dim)

        # Remember master seed
        self.seed = seed
        self.env_manager.seed = seed

        # Set the rng of the env manager
        self.env_manager.rng = self.rng

    def msg_all(self, msg: str):
        """Logs msg on master, on all workers, and in dashboard_status.txt."""
        logger.info(msg)
        self.client.run(worker_log, msg)
        with self.logdir.pfile("dashboard_status.txt").open("w") as file:
            file.write(msg)

    def finished(self):
        """Whether execution is done."""
        return self.total_evals >= self.max_evals

    def save_reload_data(self):
        """Saves data necessary for a reload.

        Current reload files:
        - reload.pkl
        - reload_em.pkl
        - reload_em.pth

        Since saving may fail due to memory issues, data is first placed in
        reload-tmp.pkl. reload-tmp.pkl then overwrites reload.pkl.

        We use gin to reference emitter classes, and pickle fails when dumping
        things constructed by gin, so we use cloudpickle instead. See
        https://github.com/google/gin-config/issues/8 for more info.
        """
        logger.info("Saving reload data")

        logger.info("Saving reload-tmp.pkl")
        with self.logdir.pfile("reload-tmp.pkl").open("wb") as file:
            reload_data = {
                "rng": self.rng,
                "outer_itrs_completed": self.outer_itrs_completed,
                "total_evals": self.total_evals,
                "metrics": self.metrics,
                "overall_min_obj": self.overall_min_obj,
                "metadata_id": self.metadata_id,
                "cur_best_id": self.cur_best_id,
                "failed_levels": self.failed_levels,
            }
            if self.is_em:
                reload_data["archive"] = self.archive
            else:
                # Do not save self.archive again here even though it is set.
                reload_data["optimizer"] = self.optimizer

            cloudpickle.dump(reload_data, file)

        if self.is_em:
            logger.info("Saving reload_em-tmp.pkl and reload_em-tmp.pth")
            self.env_manager.emulation_model.save(
                self.logdir.pfile("reload_em-tmp.pkl"),
                self.logdir.pfile("reload_em-tmp.pth"))

        logger.info("Renaming tmp reload files")
        self.logdir.pfile("reload-tmp.pkl").rename(
            self.logdir.pfile("reload.pkl"))
        if self.is_em:
            self.logdir.pfile("reload_em-tmp.pkl").rename(
                self.logdir.pfile("reload_em.pkl"))
            self.logdir.pfile("reload_em-tmp.pth").rename(
                self.logdir.pfile("reload_em.pth"))

        logger.info("Finished saving reload data")

    def save_archive(self):
        """Saves dataframes of the archive.

        The archive, including solutions and metadata, is saved to
        logdir/archive/archive_{outer_itr}.pkl

        Note that the archive is saved as an ArchiveDataFrame storing common
        Python objects, so it should be stable (at least, given fixed software
        versions).
        """
        itr = self.outer_itrs_completed
        df = self.archive.as_pandas(include_solutions=True,
                                    include_metadata=True)
        df.to_pickle(self.logdir.file(f"archive/archive_{itr}.pkl"))

    def save_archive_history(self):
        """Saves the archive's history.

        We are okay with a pickle file here because there are only numpy arrays
        and Python objects, both of which are stable.
        """
        with self.logdir.pfile("archive_history.pkl").open("wb") as file:
            pkl.dump(self.archive.history(), file)

    def save_data(self):
        """Saves archive, reload data, history, and metrics if necessary.

        This method must be called at the _end_ of each outer itr. Otherwise,
        some things might not be complete. For instance, the metrics may be in
        the middle of an iteration, so when we reload, we get an error because
        we did not end the iteration.
        """
        if self.archive_save_freq is None:
            save_full_archive = False
        elif self.archive_save_freq == -1 and self.finished():
            save_full_archive = True
        elif (self.archive_save_freq > 0
              and self.outer_itrs_completed % self.archive_save_freq == 0):
            save_full_archive = True
        else:
            save_full_archive = False

        logger.info("Saving metrics")
        self.metrics.to_json(self.logdir.file("metrics.json"))

        logger.info("Saving archive history")
        self.save_archive_history()

        if save_full_archive:
            logger.info("Saving full archive")
            self.save_archive()
        if ((self.outer_itrs_completed % self.reload_save_freq == 0)
                or self.finished()):
            self.save_reload_data()
        if self.finished():
            logger.info("Saving failed levels")
            self.logdir.save_data(self.failed_levels, "failed_levels.pkl")

    def plot_metrics(self):
        """Plots metrics every self.plot_metrics_freq itrs or on final itr."""
        if (self.outer_itrs_completed % self.plot_metrics_freq == 0
                or self.finished()):
            logger.info("Metrics:\n%s", self.metrics.get_plot_text())

    def add_performance_metrics(self):
        """Calculates various performance metrics at the end of each iter."""
        df = self.archive.as_pandas(include_solutions=False)
        objs = df.batch_objectives()
        stats = self.archive.stats

        self.metrics.add(
            "Total Evals",
            self.total_evals,
            logger,
        )
        self.metrics.add(
            "Actual QD Score",
            self.env_manager.module.actual_qd_score(objs),
            logger,
        )
        self.metrics.add(
            "Archive Size",
            stats.num_elites,
            logger,
        )
        self.metrics.add(
            "Archive Coverage",
            stats.coverage,
        )
        self.metrics.add(
            "Best Objective",
            np.max(objs),
            logger,
        )
        self.metrics.add(
            "Worst Objective",
            np.min(objs),
            logger,
        )
        self.metrics.add(
            "Mean Objective",
            np.mean(objs),
            logger,
        )
        self.metrics.add(
            "Overall Min Objective",
            self.overall_min_obj,
            logger,
        )

    def extract_metadata(self, r) -> dict:
        """Constructs metadata object from results of an evaluation."""
        meta = dataclasses.asdict(r)

        # Remove unwanted keys.
        none_keys = [key for key in meta if meta[key] is None]
        for key in itertools.chain(none_keys, []):
            try:
                meta.pop(key)
            except KeyError:
                pass

        meta["metadata_id"] = self.metadata_id
        self.metadata_id += 1

        return meta

    def build_emitters_and_optimizer(self, archive):
        """Builds pyribs components with the config params and given archive."""
        # Makes sense to initialize at zero since these are latent vectors.
        initial_solution = np.zeros(self.sol_size)

        emitters = []
        for emitter_class, n_emitters in self.emitter_types:
            emitter_seeds = self.rng.integers(np.iinfo(np.int32).max / 2,
                                              size=n_emitters,
                                              endpoint=True)
            emitters.extend([
                emitter_class(archive, initial_solution, seed=s)
                for s in emitter_seeds
            ])
            logger.info("Constructed %d emitters of class %s - seeds %s",
                        n_emitters, emitter_class, emitter_seeds)
        logger.info("Emitters: %s", emitters)

        optimizer = Optimizer(archive, emitters)
        logger.info("Optimizer: %s", optimizer)

        return emitters, optimizer

    def build_emulation_archive(self) -> ArchiveBase:
        """Builds an archive which optimizes the emulation model."""
        logger.info("Setting up pyribs components")
        seed = self.rng.integers(np.iinfo(np.int32).max / 2, endpoint=True)

        archive: ArchiveBase = self.archive_type(seed=seed,
                                                 dtype=np.float32,
                                                 record_history=False)
        logger.info("Archive: %s", archive)

        _, optimizer = self.build_emitters_and_optimizer(archive)

        for inner_itr in range(1, self.inner_itrs + 1):
            self.em_evaluate(optimizer)

            if inner_itr % 1000 == 0 or inner_itr == self.inner_itrs:
                logger.info("Completed inner iteration %d", inner_itr)

        logger.info("Generated emulation archive with %d elites (%f coverage)",
                    archive.stats.num_elites, archive.stats.coverage)

        # Save surrogate archive
        if self.save_surrogate_archive:
            save_dir = self.logdir.dir("surrogate_archive", touch=True)

            df = archive.as_pandas(include_solutions=True,
                                   include_metadata=True)
            df.to_pickle(f"{save_dir}/archive_{self.outer_itrs_completed}.pkl")

        # In downsampling, we create a smaller archive where the elite in each
        # cell is sampled from a corresponding region of cells in the main
        # archive.
        if self.downsample_em:
            downsample_archive: ArchiveBase = self.downsample_archive_type(
                seed=seed, dtype=np.float32, record_history=False)
            downsample_archive.initialize(archive.solution_dim)
            scales = np.array(archive.dims) // np.array(downsample_archive.dims)

            # Iterate through every index in the downsampled archive.
            for downsample_idx in itertools.product(
                    *map(range, downsample_archive.dims)):

                # In each index, retrieve the corresponding elites in the main
                # archive.
                elites = []
                archive_ranges = [
                    range(scales[i] * downsample_idx[i],
                          scales[i] * (downsample_idx[i] + 1))
                    for i in range(archive.behavior_dim)
                ]
                for idx in itertools.product(*archive_ranges):
                    # pylint: disable = protected-access
                    if archive._occupied[idx]:
                        elites.append(
                            Elite(archive._solutions[idx],
                                  archive._objective_values[idx],
                                  archive._behavior_values[idx], idx,
                                  archive._metadata[idx]))

                # Choose one of the elites to insert into the archive.
                if len(elites) > 0:
                    sampled_elite = elites[self.rng.integers(len(elites))]
                    downsample_archive.add(sampled_elite.sol, sampled_elite.obj,
                                           sampled_elite.beh,
                                           sampled_elite.meta)

            archive = downsample_archive

            # Save downsampled surrogate archive
            if self.save_surrogate_archive:
                save_dir = self.logdir.dir("surrogate_archive", touch=True)
                df = archive.as_pandas(include_solutions=True,
                                       include_metadata=True)
                df.to_pickle(
                    f"{save_dir}/downsample_archive_{self.outer_itrs_completed}.pkl"
                )

            logger.info(
                "Downsampled emulation archive has %d elites (%f coverage)",
                archive.stats.num_elites, archive.stats.coverage)

        return archive

    def em_evaluate(self, optimizer):
        """
        Asks for solutions from the optimizer, evaluates using the emulation
        model, and tells the objective and measures
        Args:
            optimizer: Optimizer to use
        """
        sols, _ = optimizer.ask()
        map_comps, objs, measures, success_mask = \
            self.env_manager.emulation_pipeline(sols)

        all_objs = np.full(len(map_comps), np.nan)
        all_measures = np.full((len(map_comps), self.archive.behavior_dim),
                               np.nan)
        all_objs[success_mask] = objs
        all_measures[success_mask] = measures

        # Need to add map_comps to metadata
        optimizer.tell(all_objs,
                       all_measures,
                       success_mask=success_mask,
                       metadata=map_comps)

        return sols, map_comps, objs, measures

    def evaluate_solutions(self, sols, parent_sols=None):
        """Evaluates a batch of solutions and adds them to the archive."""
        logger.info("Evaluating solutions")

        skipped_sols = 0
        if self.total_evals + len(sols) > self.max_evals:
            remaining_evals = self.max_evals - self.total_evals
            remaining_sols = remaining_evals
            skipped_sols = len(sols) - remaining_sols
            sols = sols[:remaining_sols]
            if parent_sols is not None:
                parent_sols = parent_sols[:remaining_sols]
            logger.info(
                "Unable to evaluate all solutions; will evaluate %d instead",
                remaining_sols,
            )

        logger.info("total_evals (old): %d", self.total_evals)
        self.total_evals += len(sols)
        logger.info("total_evals (new): %d", self.total_evals)

        logger.info("Distributing evaluations")

        results = self.env_manager.eval_pipeline(
            sols,
            parent_sols=parent_sols,
            batch_idx=self.outer_itrs_completed,
        )

        if self.is_em:
            logger.info(
                "Adding solutions to main archive and emulation dataset")
        else:
            logger.info("Adding solutions to the optimizer")

        objs = []
        if not self.is_em:
            measures, metadata, success_mask = [], [], []

        for sol, r in zip(sols, results):
            if not r.failed:
                obj = r.agg_obj
                objs.append(obj)  # Always insert objs.
                meas = r.agg_measures
                meta = self.extract_metadata(r)

                if self.is_em:
                    self.archive.add(sol, obj, meas, meta)
                    self.env_manager.add_experience(sol, r)
                else:
                    measures.append(meas)
                    metadata.append(meta)
                    success_mask.append(True)
            else:
                failed_level_info = self.env_manager.add_failed_info(sol, r)
                self.failed_levels.append(failed_level_info)
                if not self.is_em:
                    objs.append(np.nan)
                    measures.append(np.full(self.archive.behavior_dim, np.nan))
                    metadata.append(None)
                    success_mask.append(False)

        # Tell results to optimizer.
        if not self.is_em:
            logger.info("Filling in null values for skipped sols: %d",
                        skipped_sols)
            for _ in range(skipped_sols):
                objs.append(np.nan)
                measures.append(np.full(self.archive.behavior_dim, np.nan))
                metadata.append(None)
                success_mask.append(False)

            self.optimizer.tell(
                objs,
                measures,
                metadata,
                success_mask=success_mask,
            )

        self.metrics.add("Mean Evaluation", np.nanmean(objs), logger)
        self.overall_min_obj = min(self.overall_min_obj, np.nanmin(objs))

    def evaluate_initial_emulation_solutions(self):
        logger.info("Evaluating initial solutions")
        initial_solutions, _ = self.env_manager.get_initial_sols(
            (self.initial_sols, self.sol_size))
        self.evaluate_solutions(initial_solutions)

    def evaluate_emulation_archive(self, emulation_archive: ArchiveBase):
        logger.info("Evaluating solutions in emulation_archive")

        if self.num_elites_to_eval is None:
            sols = [elite.sol for elite in emulation_archive]
            logger.info("%d solutions in emulation_archive", len(sols))
        else:
            num_sols = len(emulation_archive)
            sols = []
            sol_values = []
            rands = self.rng.uniform(0, 1e-8, size=num_sols)  # For tiebreak

            for i, elite in enumerate(emulation_archive):
                sols.append(elite.sol)
                if self.random_sample_em:
                    new_elite = 1
                else:
                    new_elite = int(
                        self.archive.elite_with_behavior(elite.beh).obj is None)
                sol_values.append(new_elite + rands[i])

            _, sorted_sols = zip(*sorted(
                zip(sol_values, sols), reverse=True, key=lambda x: x[0]))
            sols = sorted_sols[:self.num_elites_to_eval]
            logger.info(
                f"{np.sum(np.array(sol_values) > 1e-6)} solutions predicted to "
                f"improve.")
            logger.info(
                f"Evaluating {len(sols)} out of {num_sols} solutions in "
                f"emulation_archive")

        self.evaluate_solutions(sols)

    def execute(self):
        """Runs the entire algorithm."""
        while not self.finished():
            self.msg_all(f"----- Outer Itr {self.outer_itrs_completed + 1} "
                         f"({self.total_evals} evals) -----")
            self.metrics.start_itr()
            self.archive.new_history_gen()

            if self.is_em:
                if self.outer_itrs_completed == 0:
                    self.evaluate_initial_emulation_solutions()
                else:
                    logger.info("Running inner loop")
                    self.env_manager.em_train()
                    emulation_archive = self.build_emulation_archive()
                    self.evaluate_emulation_archive(emulation_archive)
            else:
                logger.info("Running classic pyribs")
                sols, parent_sols = self.optimizer.ask()
                self.evaluate_solutions(sols, parent_sols=parent_sols)

            # Restart worker to clean up memory leak
            self.client.restart()

            logger.info("Outer itr complete - now logging and saving data")
            self.outer_itrs_completed += 1
            self.add_performance_metrics()
            self.metrics.end_itr()
            self.plot_metrics()
            self.save_data()  # Keep at end of loop (see method docstring).

        self.msg_all(f"----- Done! {self.outer_itrs_completed} itrs, "
                     f"{self.total_evals} evals -----")

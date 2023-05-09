import copy
from typing import Optional

import gin
import numpy as np
import ribs
from ribs.emitters import EmitterBase


@gin.configurable(denylist=["archive", "x0", "seed"])
class MapElitesBaselineEmitter(EmitterBase):
    """Implementation of MAP-Elites which generates solutions corresponding to
    map layout.

    Args:
        archive: Archive to store the solutions.
        x0: Initial solution. Only used for solution_dim.
        bounds: Bounds of the solution space. Pass None to
            indicate there are no bounds. Alternatively, pass an array-like to
            specify the bounds for each dim. Each element in this array-like can
            be None to indicate no bound, or a tuple of
            ``(lower_bound, upper_bound)``, where ``lower_bound`` or
            ``upper_bound`` may be None to indicate no bound. (default: None)
        seed: Random seed. (default None)
        num_objects: Solutions will be generated as ints between
            [0, num_objects)
        batch_size: Number of solutions to return in :meth:`ask`.
        initial_population: Size of the initial population before starting to
            mutate elites from the archive.
        mutation_k: Number of positions in the solution to mutate. Should be
            less than solution_dim.
        geometric_k: Whether to vary k geometrically. If it is True,
            `mutation_k` will be ignored.
        max_n_shelf: max number of shelves(index 1).
        min_n_shelf: min number of shelves(index 1).
    """
    def __init__(
        self,
        archive: ribs.archives.ArchiveBase,
        x0: np.ndarray,
        bounds: Optional["array-like"] = None,  # type: ignore
        seed: int = None,
        num_objects: int = gin.REQUIRED,
        batch_size: int = gin.REQUIRED,
        initial_population: int = gin.REQUIRED,
        mutation_k: int = gin.REQUIRED,
        geometric_k: bool = gin.REQUIRED,
        max_n_shelf: float = gin.REQUIRED,
        min_n_shelf: float = gin.REQUIRED,
    ):
        solution_dim = len(x0)
        super().__init__(archive, solution_dim, bounds)
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size

        self.num_objects = num_objects
        self.initial_population = initial_population
        self.mutation_k = mutation_k
        self.geometric_k = geometric_k
        self.max_n_shelf = max_n_shelf
        self.min_n_shelf = min_n_shelf

        if not self.geometric_k:
            assert solution_dim >= self.mutation_k

        # When we know the exact number of shelves and we only have shelf or
        # floor, k will be used to switch randomly pairs of 0's and 1's
        if self.max_n_shelf == self.min_n_shelf and self.num_objects == 2:
            if not self.geometric_k:
                assert self.min_n_shelf >= self.mutation_k
                assert self.solution_dim - self.min_n_shelf >= self.mutation_k

        self.sols_emitted = 0

    def ask(self):
        if self.sols_emitted < self.initial_population:
            if self.num_objects == 2:
                # If we know the exact number of shelves and we have one only
                # two objects (floor or shelf), we can generate solutions
                # directly.
                if self.min_n_shelf == self.max_n_shelf:
                    n_shelf = self.min_n_shelf
                    idx_array = np.tile(np.arange(self.solution_dim),
                                        (self.batch_size, 1))
                    shelf_idxs = self.rng.permuted(idx_array,
                                                   axis=1)[:, :n_shelf]
                    sols = np.zeros((self.batch_size, self.solution_dim),
                                    dtype=int)
                    for i in range(self.batch_size):
                        sols[i, shelf_idxs[i]] = 1
                    assert np.sum(sols) == self.batch_size * n_shelf
                else:
                    # If we still have only 2 objects, we can generate
                    # solutions in a biased fashion and keep generate until we
                    # have a the specified number of shelves.
                    if self.num_objects == 2:
                        sols = []
                        for _ in range(self.batch_size):
                            # Keep generate new solutions until we get desired
                            # number of shelves
                            sol = np.ones(self.solution_dim, dtype=int)
                            while not (self.min_n_shelf <= np.sum(sol) <=
                                       self.max_n_shelf):
                                sol = self.rng.choice(
                                    np.arange(self.num_objects),
                                    size=self.solution_dim,
                                    p=[
                                        1 -
                                        self.max_n_shelf / self.solution_dim,
                                        self.max_n_shelf / self.solution_dim
                                    ],
                                )
                            sols.append(sol)
            # If we have more than 2 objects, we just generate new
            # solutions directly
            else:
                sols = self.rng.integers(self.num_objects,
                                         size=(self.batch_size,
                                               self.solution_dim))

            self.sols_emitted += self.batch_size
            return np.array(sols), None

        # Mutate current solutions
        else:
            if self.num_objects == 2:
                sols = []
                parent_sols = []

                for i in range(self.batch_size):
                    if self.min_n_shelf == self.max_n_shelf:
                        parent_sol, _, _, _, meta = self.archive.get_random_elite(
                        )
                        sol = copy.deepcopy(parent_sol.astype(int))

                        # Randomly select k zero/one index pairs
                        zero_idx = np.where(sol == 0)[0]
                        one_idx = np.where(sol == 1)[0]

                        # Sample current k if it is varied geometrically
                        max_k = np.min([zero_idx.shape[0], one_idx.shape[0]])
                        curr_k = self.sample_k(max_k)

                        # Exchange black and white
                        mutate_zero_idx = self.rng.choice(zero_idx,
                                                          size=curr_k,
                                                          replace=False)
                        mutate_one_idx = self.rng.choice(one_idx,
                                                         size=curr_k,
                                                         replace=False)

                        # Swap 0's and 1's
                        sol[mutate_zero_idx] = 1
                        sol[mutate_one_idx] = 0
                        assert np.sum(
                            sol) == self.min_n_shelf == self.max_n_shelf

                    else:
                        # Start with all 1's so that we get into the while loop.
                        sol = np.ones(self.solution_dim)

                        # keep resampling until we have desired number of
                        # shelves
                        while not (self.min_n_shelf <= np.sum(sol) <=
                                   self.max_n_shelf):
                            # Sample current k
                            curr_k = self.sample_k(self.solution_dim)

                            # Select k spots randomly without replacement
                            # and calculate the random replacement values
                            idx_array = np.arange(self.solution_dim)
                            mutate_idxs = self.rng.permuted(idx_array)[:curr_k]
                            mutate_vals = self.rng.integers(self.num_objects,
                                                            size=curr_k)
                            parent_sol, _, _, _, meta = \
                                self.archive.get_random_elite()

                            sol = copy.deepcopy(parent_sol.astype(int))
                            # Replace with mutated values
                            for j in range(curr_k):
                                sol[mutate_idxs[j]] = mutate_vals[j]
                    sols.append(sol)

                    # Get the repaired parent sol
                    if meta is not None and "warehouse_metadata" in meta:
                        parent_sols.append(
                            meta["warehouse_metadata"]["map_int"])

            # We have more than 2 objects
            else:
                sols = []
                parent_sols = []

                # select k spots randomly without replacement
                # and calculate the random replacement values
                curr_k = self.sample_k(self.solution_dim)
                idx_array = np.tile(np.arange(self.solution_dim),
                                    (self.batch_size, 1))
                mutate_idxs = self.rng.permuted(idx_array, axis=1)[:, :curr_k]
                mutate_vals = self.rng.integers(self.num_objects,
                                                size=(self.batch_size, curr_k))

                for i in range(self.batch_size):
                    parent_sol, _, _, _, meta = self.archive.get_random_elite()
                    sol = copy.deepcopy(parent_sol.astype(int))
                    # Replace with random values
                    sol[mutate_idxs[i]] = mutate_vals[i]
                    sols.append(sol)

                    # Get the repaired parent sol
                    if meta is not None and "warehouse_metadata" in meta:
                        parent_sols.append(
                            meta["warehouse_metadata"]["map_int"])

            self.sols_emitted += self.batch_size
            return np.array(sols), np.array(parent_sols)

    def sample_k(self, max_k):
        if self.geometric_k:
            curr_k = self.rng.geometric(p=0.5)
            # Clip k if necessary
            if curr_k > max_k:
                curr_k = max_k
        else:
            curr_k = self.mutation_k
        return curr_k
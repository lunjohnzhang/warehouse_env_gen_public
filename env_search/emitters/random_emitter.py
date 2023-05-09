import copy
from typing import Optional

import gin
import numpy as np
import ribs
from ribs.emitters import EmitterBase


@gin.configurable(denylist=["archive", "x0", "seed"])
class RandomEmitter(EmitterBase):
    """Implementation of Random Search which generates random solutions.

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
    """

    def __init__(self,
                 archive: ribs.archives.ArchiveBase,
                 x0: np.ndarray,
                 bounds: Optional["array-like"] = None, # type: ignore
                 seed: int = None,
                 num_objects: int = gin.REQUIRED,
                 batch_size: int = gin.REQUIRED,
                 p_obstacle: float = gin.REQUIRED,):
        solution_dim = len(x0)
        super().__init__(archive, solution_dim, bounds)
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.num_objects = num_objects
        self.p_obstacle = p_obstacle

        self.sols_emitted = 0

    def ask(self):
        self.sols_emitted += self.batch_size
        sols = []
        for _ in range(self.batch_size):
            # Keep generate new solutions until we get low enough p_obstacle
            sol = np.ones(self.solution_dim)
            while np.sum(sol) / self.solution_dim >= self.p_obstacle:
                sol = self.rng.choice(
                    np.arange(self.num_objects),
                    size=self.solution_dim,
                    p=[1 - self.p_obstacle, self.p_obstacle],
                )
            sols.append(sol)
        return np.array(sols)

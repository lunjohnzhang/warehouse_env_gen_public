"""Custom GridArchive."""
import gin
import numpy as np
import ribs.archives
from ribs.archives._archive_base import readonly


@gin.configurable
class GridArchive(ribs.archives.GridArchive):
    """Based on pyribs GridArchive.

    This archive records history of its objectives and behavior values if
    record_history is True. Before each generation, call new_history_gen() to
    start recording history for that gen. new_history_gen() must be called
    before calling add() for the first time.
    """

    def __init__(self,
                 dims,
                 ranges,
                 seed=None,
                 dtype=np.float64,
                 record_history=True):
        super().__init__(dims, ranges, seed, dtype)
        self._record_history = record_history
        self._history = [] if self._record_history else None

    def best_elite(self):
        """Returns the best Elite in the archive."""
        if self.empty:
            raise IndexError("No elements in archive.")

        objectives = self._objective_values[self._occupied_indices_cols]
        idx = self._occupied_indices[np.argmax(objectives)]
        return ribs.archives.Elite(
            readonly(self._solutions[idx]),
            self._objective_values[idx],
            readonly(self._behavior_values[idx]),
            idx,
            self._metadata[idx],
        )

    def new_history_gen(self):
        """Starts a new generation in the history."""
        if self._record_history:
            self._history.append([])

    def history(self):
        """Gets the current history."""
        return self._history

    def add(self, solution, objective_value, behavior_values, metadata=None):
        status, val = super().add(solution, objective_value, behavior_values,
                                  metadata)

        # Only save obj and BCs in the history.
        if self._record_history and status:
            self._history[-1].append([objective_value, behavior_values])

        return status, val

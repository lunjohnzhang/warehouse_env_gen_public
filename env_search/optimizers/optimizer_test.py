"""Tests extra features of the optimizer."""
import numpy as np

from src.archives import GridArchive
from src.emitters import GaussianEmitter
from src.optimizers import Optimizer


def test_no_success_mask():
    solution_dim = 2
    num_solutions = 2
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)], record_history=False)
    emitters = [
        GaussianEmitter(archive, [0.0, 0.0],
                        solution_dim,
                        batch_size=num_solutions),
        GaussianEmitter(archive, [0.0, 0.0],
                        solution_dim,
                        batch_size=num_solutions),
    ]
    opt = Optimizer(archive, emitters)

    _ = opt.ask()
    opt.tell(
        [1, 1, 1, 1],
        [[-1, -1], [-1, 1], [1, -1], [1, 1]],
    )
    assert len(archive) == 4


def test_tell_success_mask():
    solution_dim = 2
    num_solutions = 2
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)], record_history=False)
    emitters = [
        GaussianEmitter(archive, [0.0, 0.0],
                        solution_dim,
                        batch_size=num_solutions),
        GaussianEmitter(archive, [0.0, 0.0],
                        solution_dim,
                        batch_size=num_solutions),
    ]
    opt = Optimizer(archive, emitters)

    _ = opt.ask()
    opt.tell(
        [1, -np.inf, -np.inf, 1],
        [[-1, -1], [0, 0], [0, 0], [1, 1]],
        success_mask=[True, False, False, True],
    )
    assert len(archive) == 2

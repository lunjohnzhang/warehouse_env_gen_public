"""Tests for GridArchive."""
import numpy as np

from src.archives import GridArchive


def test_history():
    archive = GridArchive([10, 10], [(-1, 1)] * 2)
    archive.initialize(2)

    archive.new_history_gen()
    archive.add([1, 2], 1.0, [0.5, 0.5])
    archive.add([1, 2], 0.0, [0.5, 0.5])
    archive.add([1, 2], 2.0, [0.5, 0.5])

    archive.new_history_gen()
    archive.add([1, 2], 2.0, [0.1, 0.1])
    archive.add([1, 2], 3.0, [0.1, 0.1])
    archive.add([1, 2], 4.0, [0.1, 0.1])

    h = archive.history()

    # Check h[0].

    assert len(h[0]) == 2

    assert h[0][0][0] == 1.0
    assert np.all(h[0][0][1] == [0.5, 0.5])

    assert h[0][1][0] == 2.0
    assert np.all(h[0][1][1] == [0.5, 0.5])

    # Check h[1].

    assert len(h[1]) == 3

    assert h[1][0][0] == 2.0
    assert np.all(h[1][0][1] == [0.1, 0.1])

    assert h[1][1][0] == 3.0
    assert np.all(h[1][1][1] == [0.1, 0.1])

    assert h[1][2][0] == 4.0
    assert np.all(h[1][2][1] == [0.1, 0.1])

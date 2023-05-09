"""Benchmarks the metric logger.

We had some issues with get_plot_text() being slow, but those have since been
resolved.

Usage:
    pyinstrument -m src.utils.metric_logger_benchmark
"""
import time

import fire
import numpy as np

from src.utils.metric_logger import MetricLogger


def benchmark(n_series: int = 20, n_points: int = 4000):
    """Adds and plots n_series metrics with n_points each."""
    start_time = time.time()

    m = MetricLogger([(str(i), i % 2 == 0) for i in range(n_series)])

    # Random points in the range [0, 10k).
    data = np.random.random((n_series, n_points)) * 10000

    # Add all the data to the logger.
    print("Adding Data...")
    for j in range(n_points):
        m.start_itr()
        for i in range(n_series):
            m.add(str(i), data[i, j])
        m.end_itr()
    print("Added all data")

    print("Plots:", m.get_plot_text())

    print("Time:", time.time() - start_time)


if __name__ == "__main__":
    fire.Fire(benchmark)

"""Provides a utility for logging metrics across iterations."""
import json
import logging
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from env_search.utils.text_plot import text_plot


class MetricLogger:
    """Tracks various pieces of scalar metrics across generations.

    Two metrics are automatically tracked here. They are the time per iteration
    (see MetricLogger.ITR_TIME) and cumulative time (see MetricLogger.CUM_TIME).

    Args:
        metric_list: A list of metric descriptions. Each description is a tuple
            of (name, use_zero) where use_zero tells whether the metric starts
            at a value of 0 on iteration 0 (some values, such as itr time, do
            not make sense if they start at 0, while others, like archive size,
            do make sense).
    Usage:
        metrics = MetricLogger()
        for itr in range(iterations):
            metrics.start_itr()
            metrics.add("metric 1", value)
            metrics.add("metric 2", value)
            metrics.end_itr()
            metrics.display_text()
    """

    ITR_TIME = "Itr Time"
    CUM_TIME = "Cum Time"

    def __init__(self, metric_list: List[Tuple[str, bool]]):
        self._active_itr = False
        self._total_itrs = 0
        self._itr_start_time = None
        self._added_this_itr = None

        metric_list = [(self.CUM_TIME, True), (self.ITR_TIME, False),
                       *metric_list]
        self._metrics = OrderedDict()
        for metric_name, use_zero in metric_list:
            self._metrics[metric_name] = {
                "data": [0] if use_zero else [],
                "use_zero": use_zero,
            }

    @property
    def names(self) -> List[str]:
        """List of names of metrics in this logger."""
        return list(self._metrics.keys())

    @property
    def total_itrs(self) -> int:
        """Total number of iterations completed so far."""
        return self._total_itrs

    def to_json(self, jsonfile: str):
        """Saves the logger's info in JSON format.

        Args:
            jsonfile: Name of the file to save to.
        """
        with open(jsonfile, "w") as file:
            json.dump(
                {
                    "active_itr": self._active_itr,
                    "total_itrs": self._total_itrs,
                    "itr_start_time": self._itr_start_time,
                    "added_this_itr": list(self._added_this_itr),
                    "metrics": self._metrics,
                },
                file,
                indent=2,
            )

    @staticmethod
    def from_json(jsonfile: str):
        """Constructs a logger from the data in the JSON file.

        Args:
            jsonfile: Name of the file to load from.
        """
        with open(jsonfile, "r") as file:
            data = json.load(file)

        # pylint: disable = protected-access
        metrics = MetricLogger([])
        metrics._active_itr = data["active_itr"]
        metrics._total_itrs = data["total_itrs"]
        metrics._itr_start_time = data["itr_start_time"]
        metrics._added_this_itr = set(data["added_this_itr"])
        metrics._metrics = OrderedDict(data["metrics"])

        return metrics

    def start_itr(self):
        """Starts the iteration."""
        if self._active_itr:
            raise RuntimeError("Already in the middle of an iteration.")
        self._active_itr = True
        self._added_this_itr = set()
        self._itr_start_time = time.time()

    def end_itr(self, itr_time: Optional[float] = None):
        """Ends the iteration.

        Upon ending the iteration, iteration and cumulative time are also
        recorded.

        Args:
            itr_time: Forces the iteration time to be a certain value, instead
                of the actual amount of time that elapsed since calling
                start_itr().
        Raises:
            RuntimeError: This method was called without calling start_itr().
            RuntimeError: Not all metrics were added before calling this method.
        """
        if not self._active_itr:
            raise RuntimeError(
                "Iteration has not been started. Call start_itr().")
        self._active_itr = False
        self._total_itrs += 1

        # Check whether all metrics were added.
        remaining_metrics = (set(self._metrics.keys()) - self._added_this_itr -
                             {self.ITR_TIME, self.CUM_TIME})
        if len(remaining_metrics) > 0:
            raise RuntimeError("The following metrics were not added this "
                               f"itr: {remaining_metrics}")

        # Handle time.
        itr_time = (itr_time if itr_time is not None else time.time() -
                    self._itr_start_time)
        self._metrics[self.ITR_TIME]["data"].append(itr_time)
        self._metrics[self.CUM_TIME]["data"].append(
            self._metrics[self.CUM_TIME]["data"][-1] + itr_time)

    def add(self,
            name: str,
            value: Union[float, int],
            logger: logging.Logger = None):
        """Adds the given metric.

        Args:
            name: The name of the metric. This must be one of the metrics
                provided in the constructor.
            value: the scalar value to log.
            logger: If not None, this logger will be used to log the metric to
                the console immediately.
        Raises:
            RuntimeError: The metric name is not recognized.
            RuntimeError: The metric has already been added this itr.
        """
        if not self._active_itr:
            raise RuntimeError(
                "Iteration has not been started. Call start_itr().")
        if name not in self._metrics:
            raise RuntimeError(f"Unknown metric '{name}'")
        if name in self._added_this_itr:
            raise RuntimeError(f"Metric '{name}' already added this itr")

        # Convert to Python types.
        if isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, np.integer):
            value = int(value)

        self._metrics[name]["data"].append(value)
        self._added_this_itr.add(name)
        if logger is not None:
            logger.info("%s: %s", name, str(value))

    def add_post(self, name: str, values: Union[Sequence[float], Sequence[int]],
                 use_zero: bool):
        """Add a new metric that was not given at initialization.

        This method cannot be called in the middle of an iteration.

        Raises:
            RuntimeError: Iteration is currently active.
            ValueError: values is of the wrong length.
        """
        if self._active_itr:
            raise RuntimeError("Call end_itr() before calling this method.")

        expected_length = self._total_itrs + int(use_zero)
        if len(values) != expected_length:
            raise ValueError(f"values should be length {expected_length} but "
                             f"is length {len(values)}")

        self._metrics[name] = {
            "data": list(values),
            "use_zero": use_zero,
        }

    def remove(self, name: str):
        """Removes a metric if it exists."""
        self._metrics.pop(name, None)

    def get_plot_data(self) -> Dict:
        """Returns the data in a form suitable for plotting metrics vs. itrs.

        Specifically, the data looks like this::

            {
                name: {
                    "x": [0, 1, 2, ...] # 0 may be excluded.
                    "y": [...] # metric values.
                }
                ... # More metrics
            }

        Note this method will only work when not in an active iteration, as data
        may be updated during an iteration.

        Returns:
            See above.
        Raises:
            RuntimeError: Iteration is currently active.
        """
        if self._active_itr:
            raise RuntimeError("Call end_itr() before calling this method.")
        data = {}
        x_with_zero = list(range(self._total_itrs + 1))
        x_no_zero = list(range(1, self._total_itrs + 1))
        for name in self._metrics:
            data[name] = {
                "x": (x_with_zero
                      if self._metrics[name]["use_zero"] else x_no_zero),
                "y": self._metrics[name]["data"],
            }
        return data

    def get_single(self, name: str) -> Dict:
        """Returns the data for plotting one metric.

        Args:
            name: Name of the metric to retrieve.
        Returns:
            Dict with plot data for the given metric. Equivalent to one of the
            entries in get_plot_data().
        Raises:
            IndexError: name is not a valid metric.
        """
        if name not in self._metrics:
            raise IndexError(f"'{name}' is not a known metric")
        return {
            "x":
                list(
                    range(int(not self._metrics[name]["use_zero"]),
                          self._total_itrs + 1)),
            "y":
                self._metrics[name]["data"],
        }

    def get_plot_text(self, plot_width: int = 80, plot_height: int = 20) -> str:
        """Generates string with plots of all the data.

        Args:
            plot_width: Width of each plot in characters.
            plot_height: Height of each plot in characters.
        Returns:
            A multi-line string with all the plots joined together.
        """
        data = self.get_plot_data()
        output = []
        for name, array in data.items():
            output.extend([
                f"=== {name} (Last val: {array['y'][-1]}) ===",
                text_plot(array["x"], array["y"], plot_width, plot_height)
            ])
        return "\n".join(output)

    def display_text(self, plot_width: int = 80, plot_height: int = 20):
        """Print out all the plots."""
        print(self.get_plot_text(plot_width, plot_height))

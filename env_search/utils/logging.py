"""Utilities for logging."""
import logging
import sys

from dask.distributed import get_worker


def setup_logging(on_worker=True, level: int = logging.INFO):
    """Sets up the logger to write to stdout."""
    worker_str = f"({get_worker().name}) " if on_worker else ""
    log_format = (f"{worker_str}[%(levelname)s|%(asctime)s|%(name)s:%(lineno)d]"
                  " %(message)s")
    logging.basicConfig(format=log_format, level=level, stream=sys.stdout)


def worker_log(msg: str, level: int = logging.INFO):
    """Logs a message on the worker.

    Intended to be used to run a logging message on all workers with
    dask.distributed.Client.run, e.g.

        client.run(worker_log, "this is a message")
    """
    logger = logging.getLogger(__name__)
    logger.log(level, msg)

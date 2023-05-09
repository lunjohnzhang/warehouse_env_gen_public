from math import log
import os
import gin
import fire
import json
import glob
import dask
import torch
import shutil
import logging

from pathlib import Path
from logdir import LogDir
from typing import Optional, Union
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from env_search.manager import Manager
from env_search.utils.logging import setup_logging
from dask import config as cfg

cfg.set({'distributed.scheduler.worker-ttl': None})


def setup_logdir(seed: int,
                 slurm_logdir: Union[str, Path],
                 reload_dir: Optional[str] = None):
    """Creates the logging directory with a LogDir object.

    Args:
        seed: Master seed.
        slurm_logdir: Directory for storing Slurm logs. Pass None if not
            applicable.
        reload_dir: Directory for reloading. If passed in, this directory will
            be reused as the logdir.
    """
    name = gin.query_parameter("experiment.name")

    if reload_dir is not None:
        # Reuse existing logdir.
        reload_dir = Path(reload_dir)
        logdir = LogDir(name, custom_dir=reload_dir)
    else:
        # Create new logdir.
        logdir = LogDir(name, rootdir="./logs", uuid=True)

    # Save configuration options.
    with logdir.pfile("config.gin").open("w") as file:
        file.write(gin.config_str(max_line_length=120))

    # Write a README.
    logdir.readme(git_commit=False, info=[f"Seed: {seed}"])

    # Write the seed.
    with logdir.pfile("seed").open("w") as file:
        file.write(str(seed))

    if slurm_logdir is not None:
        # Write the logging directory to the slurm logdir.
        with (Path(slurm_logdir) / "logdir").open("w") as file:
            file.write(str(logdir.logdir))

        # Copy the hpc config.
        hpc_config = glob.glob(str(Path(slurm_logdir) / "config" / "*.sh"))[0]
        hpc_config_copy = logdir.file("hpc_config.sh")
        shutil.copy(hpc_config, hpc_config_copy)

    return logdir


def check_env():
    """Environment check(s)."""
    assert os.environ['OPENBLAS_NUM_THREADS'] == '1', \
        ("OPENBLAS_NUM_THREADS must be set to 1 so that the numpy in each "
         "worker does not throttle each other. If you are running in the "
         "Singularity container, this should already be set.")


@gin.configurable(denylist=["client", "logdir", "seed", "reload"])
def experiment(client: Client,
               logdir: LogDir,
               seed: int,
               reload: bool = False,
               name: str = gin.REQUIRED):
    """Executes a distributed experiment on Dask.

    Args:
        client: A Dask client for running distributed tasks.
        logdir: A logging directory instance for recording info.
        seed: Master seed for the experiment.
        reload: Whether to reload experiment from logdir.
        name: Name of the experiment.
    """
    logging.info("Experiment Name: %s", name)

    # All further configuration to Manager is handled by gin.
    Manager(
        client=client,
        logdir=logdir,
        seed=seed,
        reload=reload,
    ).execute()


def main(
    config: str,
    seed: int = 0,
    address: str = "127.0.0.1:8786",
    reload: str = None,
    slurm_logdir=None,
):
    gin.parse_config_file(config)
    check_env()

    # Set up logdir
    logdir = setup_logdir(seed, slurm_logdir, reload)

    # Set up dask
    # client = setup_dask(logdir=logdir)
    client = Client(address)

    # Set up logging
    setup_logging(on_worker=False)
    client.register_worker_callbacks(setup_logging)

    # On the workers, PyTorch is entirely CPU-based. Since we run multiple
    # processes on each cluster node, allowing PyTorch to be multithreaded would
    # result in race conditions and thus slow down the code. This is similar to
    # how we force numpy and OpenBLAS to be single-threaded.
    client.register_worker_callbacks(lambda: torch.set_num_threads(1))

    # We wait for at least one worker to join the cluster before doing anything,
    # as methods like client.scatter fail when there are no workers.
    logging.info("Waiting for at least 1 worker to join cluster")
    client.wait_for_workers(1)
    logging.info("At least one worker has joined")

    logdir.save_data(client.ncores(), "client.json")
    logging.info("Dask dashboard: %s", client.dashboard_link)
    logging.info("Master Seed: %d", seed)
    logging.info("Logging Directory: %s", logdir.logdir)
    logging.info("CPUs: %s", client.ncores())
    logging.info("===== Config: =====\n%s", gin.config_str())

    experiment(
        client=client,
        logdir=logdir,
        seed=seed,
        reload=reload is not None,
    )


if __name__ == '__main__':
    fire.Fire(main)

# Multi-Robot Coordination and Layout Design for Automated Warehousing

This repository is the official implementation of **[Multi-Robot Coordination and Layout Design for Automated Warehousing](https://arxiv.org/abs/2305.06436)** published in IJCAI 2023. The repository builds on top of the repositories of [Deep Surrogate Assisted Generation of Environment (DSAGE)](https://github.com/icaros-usc/dsage) and [Rolling-Horizon Collision Resolution (RHCR)](https://github.com/Jiaoyang-Li/RHCR).

## Installation

This is a hybrid C++/Python project. The simulation environment is written in C++ and the rests are in Python. We use [pybind11](https://pybind11.readthedocs.io/en/stable/) to bind the two languages.

1. **Initialize pybind11:** After cloning the repo, initialize the pybind11 submodule

   ```bash
   git submodule init
   git submodule update
   ```

2. **Install Singularity:** All of our code runs in a Singularity container.
   Singularity is a container platform (similar in many ways to Docker). Please
   see the instructions
   [here](https://sylabs.io/guides/3.6/user-guide/quick_start.html) for
   installing Singularity 3.6.

3. **Download Boost:** From the root directory of the project, run the following to download the Boost 1.71, which is required for compiling C++ simulator. You don't have to install it on your system since it will be passed into the container and installed there.

   ```
   wget https://boostorg.jfrog.io/artifactory/main/release/1.71.0/source/boost_1_71_0.tar.gz --no-check-certificate
   ```

4. **Install CPLEX:** CPLEX is used for repairing the generated warehouse maps.

   1. Download the free academic version [here](https://www.ibm.com/products/ilog-cplex-optimization-studio).
   2. Download the installation file for Linux.
   3. Follow this [guide](https://www.ibm.com/docs/en/icos/12.10.0?topic=v12100-installing-cplex-optimization-studio) to install it. Basically:

   ```
   chmod u+x INSTALLATION_FILE
   ./INSTALLATION_FILE
   ```

   During installation, set the installation directory to `CPLEX_Studio2210/` in the repo.

5. **Build Singularity container:** Run the provided script to build the container. Note that this need `sudo` permission on your system.
   ```
   bash build_container.sh
   ```
   The script will first build a container as a sandbox, compile the C++ simulator, then convert that to a regular `.sif` Singularity container.

## Instructions

### Logging Directory Manifest

Regardless of where the script is run, the log files and results are placed in a
logging directory in `logs/`. The directory's name is of the form
`%Y-%m-%d_%H-%M-%S_<dashed-name>_<uuid>`, e.g.
`2020-12-01_15-00-30_experiment-1_ff1dcb2b`. Inside each directory are the
following files:

```text
- config.gin  # All experiment config variables, lumped into one file.
- seed  # Text file containing the seed for the experiment.
- reload.pkl  # Data necessary to reload the experiment if it fails.
- reload_em.pkl  # Pickle data for EmulationModel.
- reload_em.pth  # PyTorch models for EmulationModel.
- metrics.json  # Data for a MetricLogger with info from the entire run, e.g. QD score.
- hpc_config.sh  # Same as the config in the Slurm dir, if Slurm is used.
- archive/  # Snapshots of the full archive, including solutions and metadata,
            # in pickle format.
- archive_history.pkl  # Stores objective values and behavior values necessary
                       # to reconstruct the archive. Solutions and metadata are
                       # excluded to save memory.
- dashboard_status.txt  # Job status which can be picked up by dashboard scripts.
                        # Only used during execution.
- evaluations # Output logs of LMAPF simulator
```

### Running Locally

#### Single Run

To run one experiment locally, use:

```bash
bash scripts/run_local.sh CONFIG SEED NUM_WORKERS
```

For instance, with 4 workers:

```bash
bash scripts/run_local.sh config/foo.gin 42 4
```

`CONFIG` is the [gin](https://github.com/google/gin-config) experiment config
for `env_search/main.py`.

### Running on Slurm

Use the following command to run an experiment on an HPC with Slurm (and
Singularity) installed:

```bash
bash scripts/run_slurm.sh CONFIG SEED HPC_CONFIG
```

For example:

```bash
bash scripts/run_slurm.sh config/foo.gin 42 config/hpc/test.sh
```

`CONFIG` is the experiment config for `env_search/main.py`, and `HPC_CONFIG` is a shell
file that is sourced by the script to provide configuration for the Slurm
cluster. See `config/hpc` for example files.

Once the script has run, it will output commands like the following:

- `tail -f ...` - You can use this to monitor stdout and stderr of the main
  experiment script. Run it.
- `bash scripts/slurm_cancel.sh ...` - This will cancel the job.

### Reloading

While the experiment is running, its state is saved to `reload.pkl` in the
logging directory. If the experiment fails, e.g. due to memory limits, time
limits, or network connection issues, `reload.pkl` may be used to continue the
experiment. To do so, execute the same command as before, but append the path to
the logging directory of the failed experiment.

```bash
bash scripts/run_slurm.sh config/foo.gin 42 config/hpc/test.sh -r logs/.../
```

The experiment will then run to completion in the same logging directory. This
works with `scripts/run_local.sh` too.

## Reproducing Paper Results

The `config/` directory contains the config files required to run the experiments shown in the paper.

| Config file                                                      | QD Search Experiment                      |
| ---------------------------------------------------------------- | ----------------------------------------- |
| config/warehouse/DSAGE_home-location_scenario_DPP.gin            | DSAGE + DPP in home-location              |
| config/warehouse/DSAGE_home-location_scenario_RHCR_40_agents.gin | DSAGE + RHCR (40 agents) in home-location |
| config/warehouse/DSAGE_home-location_scenario_RHCR_60_agents.gin | DSAGE + RHCR (60 agents) in home-location |
| config/warehouse/DSAGE_home-location_scenario_RHCR_88_agents.gin | DSAGE + RHCR (88 agents) in home-location |
| config/warehouse/MAP-Elites_home-location_scenario_DPP.gin       | MAP-Elites + DPP in home-location         |
| config/warehouse/MAP-Elites_home-location_scenario_RHCR.gin      | MAP-Elites + RHCR in home-location        |
| config/warehouse/DSAGE_small_workstation_scenario_RHCR.gin       | DSAGE + RHCR in small workstation         |
| config/warehouse/DSAGE_medium_workstation_scenario_RHCR.gin      | DSAGE + RHCR in medium workstation        |
| config/warehouse/DSAGE_large_workstation_scenario_RHCR.gin       | DSAGE + RHCR in large workstation         |
| config/warehouse/MAP-Elites_small_workstation_scenario_RHCR.gin  | MAP-Elites + RHCR in small workstation    |
| config/warehouse/MAP-Elites_medium_workstation_scenario_RHCR.gin | MAP-Elites + RHCR in medium workstation   |
| config/warehouse/MAP-Elites_large_workstation_scenario_RHCR.gin  | MAP-Elites + RHCR in large workstation    |

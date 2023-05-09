"""Visualizes heatmaps from an experiment.

Usage:
    python env_search/analysis/throughput.py --logdirs_plot <log_dir_plot>
"""
import os
import json
from typing import List

import fire
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
import scipy.stats as st

from env_search.analysis.utils import get_color, algo_name_map


mpl.use("agg")

# set matplotlib params
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def throughput_thr_time(logdirs_plot: str, ax=None):
    with open(os.path.join(logdirs_plot, "meta.yaml"), "r") as f:
        meta = yaml.safe_load(f)

    algo_name = meta["algorithm"]
    map_size = meta["map_size"]
    mode = meta["mode"]
    map_from = meta["map_from"]
    n_agents_opt = meta.get("n_agents_opt", None)

    all_per_t_throughputs = []
    all_throughput = []

    y_max = None
    if mode == "w":
        y_min = 0
        if map_size == "small":
            y_max = 9
        elif map_size == "medium":
            y_max = 9
        elif map_size == "large":
            y_max = 9
            y_min = 0
    elif mode == "r":
        y_min = 0
        if map_size == "small":
            y_max = 9
        elif map_size == "medium":
            y_max = 16
        elif map_size == "large":
            y_max = 45

    # Will loop through all log directories and take average/cf of all runs
    for logdir_f in os.listdir(logdirs_plot):
        logdir = os.path.join(logdirs_plot, logdir_f)
        if not os.path.isdir(logdir):
            continue
        results_dir = os.path.join(logdir, "results")
        for sim_dir in os.listdir(results_dir):
            sim_dir_comp = os.path.join(results_dir, sim_dir)
            config_file = os.path.join(sim_dir_comp, "config.json")
            result_file = os.path.join(sim_dir_comp, "result.json")

            with open(config_file, "r") as f:
                config = json.load(f)

            with open(result_file, "r") as f:
                result = json.load(f)

            tasks_finished_timestep = np.array(
                result["tasks_finished_timestep"])
            per_time_throughput = tasks_finished_timestep[:, 0]
            timesteps = tasks_finished_timestep[:, 1]
            all_per_t_throughputs.append(per_time_throughput)
            all_throughput.append(result["throughput"])

    # Take the average and confidence interval
    all_per_t_throughputs = np.array(all_per_t_throughputs)
    if algo_name == "RHCR":
        # Divide by time window of RHCR
        # mean_throughputs /= 5
        all_per_t_throughputs = all_per_t_throughputs / 5
    mean_throughputs = np.mean(all_per_t_throughputs, axis=0)

    cf_throughputs = st.t.interval(confidence=0.95,
                                   df=len(all_per_t_throughputs) - 1,
                                   loc=mean_throughputs,
                                   scale=st.sem(all_per_t_throughputs, axis=0))

    save_fig = False
    if ax is None:
        save_fig = True
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    color = get_color(map_from, algo_name, n_agents_opt)
    label = f"{map_from} + {algo_name_map[algo_name]}"
    if n_agents_opt is not None:
        label += f"({n_agents_opt} agents)"
    ax.plot(
        timesteps,
        mean_throughputs,
        # marker=".",
        color=color,
        label=label,
        # label=f"{map_from}",
    )
    ax.fill_between(
        timesteps,
        cf_throughputs[1],
        cf_throughputs[0],
        alpha=0.5,
        color=color,
    )

    if save_fig:
        ax.set_ylabel("Number of Finished Tasks", fontsize=25)
        ax.set_xlabel("Timestep", fontsize=25)
        ax.set_xlim(0, timesteps[-1])
        ax.set_ylim(y_min, y_max)
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=15)

        ax.figure.tight_layout()
        fig.savefig(
            os.path.join(
                logdirs_plot,
                f"throughput_thr_time_{algo_name}_{map_size}_{mode}.png",
            ),
            dpi=300,
        )

    avg_throughput = np.mean(all_throughput)
    print(f"Average Throughput of all simulations: {avg_throughput}")

    return timesteps, y_min, y_max, meta


if __name__ == "__main__":
    fire.Fire(throughput_thr_time)

"""Visualizes throughput vs. n_agents from an experiment.

Usage:
    python env_search/analysis/throughput_vs_n_agents.py --logdirs_plot <log_dir_plot>
"""
import os
import json
import warnings
from typing import List

import fire
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
import scipy.stats as st
import pandas as pd



from env_search.analysis.utils import get_color, algo_name_map

mpl.use("agg")

# set matplotlib params
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# color_map = {
#     "DSAGE": "green",
#     "MAP-Elites": "red",
#     # "ours": "red",
#     "Human": "blue",
#     "CMA-ME + NCA": "cyan",
#     "CMA-MAE + NCA": "gold",
# }


def add_item_to_dict_by_agent_num(to_dict, agent_num, element):
    if agent_num in to_dict:
        to_dict[agent_num].append(element)
    else:
        to_dict[agent_num] = [element]


def sort_and_get_vals_from_dict(the_dict):
    the_dict = sorted(the_dict.items())
    agent_nums = [agent_num for agent_num, _ in the_dict]
    all_vals = [vals for _, vals in the_dict]
    return agent_nums, all_vals


def compute_numerical(vals, all_success_vals):
    # Take the average, confidence interval and standard error
    all_vals = np.array(vals)
    all_success_vals = np.array(all_success_vals)
    assert all_vals.shape == all_success_vals.shape
    # breakpoint()
    mean_vals = np.mean(all_vals, axis=1)
    mean_vals_success = []
    sem_vals_success = []
    for i, curr_vals in enumerate(all_vals):
        # curr_vals = [x for x in curr_vals if x != 0]
        filtered_curr_vals = []
        for j, x in enumerate(curr_vals):
            if all_success_vals[i, j] == 1:
                filtered_curr_vals.append(x)
        # Supress the runtime warning about mean of empty slice and sem of 0 or
        # 1 element
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_vals_success.append(np.mean(filtered_curr_vals))
            sem_vals_success.append(st.sem(filtered_curr_vals))

    cf_vals = st.t.interval(confidence=0.95,
                            df=all_vals.shape[1] - 1,
                            loc=mean_vals,
                            scale=st.sem(all_vals, axis=1) + 1e-8)
    sem_vals = st.sem(all_vals, axis=1)
    return mean_vals, cf_vals, sem_vals, mean_vals_success, sem_vals_success


def throughput_vs_n_agents(logdirs_plot: str, ax=None):
    with open(os.path.join(logdirs_plot, "meta.yaml"), "r") as f:
        meta = yaml.safe_load(f)

    algo_name = meta["algorithm"]
    map_size = meta["map_size"]
    mode = meta["mode"]
    map_from = meta["map_from"]
    n_agents_opt = meta.get("n_agents_opt", None)
    all_throughputs_dict = {}  # Raw throughput
    all_runtime_dict = {}
    all_success_dict = {}
    y_min = 0
    y_max = 10

    for logdir_f in os.listdir(logdirs_plot):
        logdir = os.path.join(logdirs_plot, logdir_f)
        if not os.path.isdir(logdir):
            continue
        results_dir = os.path.join(logdir, "results")
        # agent_nums = []
        # throughputs = []
        for sim_dir in os.listdir(results_dir):
            sim_dir_comp = os.path.join(results_dir, sim_dir)
            config_file = os.path.join(sim_dir_comp, "config.json")
            result_file = os.path.join(sim_dir_comp, "result.json")

            if os.path.exists(config_file) and os.path.exists(result_file):

                with open(config_file, "r") as f:
                    config = json.load(f)

                with open(result_file, "r") as f:
                    result = json.load(f)

                congested = result["congested"]
                agent_num = config["agentNum"]

                # Only consider the uncongested simulations
                throughput = result["throughput"]  # if not congested else 0
                runtime = result["cpu_runtime"]  # if not congested else 0
                success = 1 if not congested else 0
                # agent_nums.append(agent_num)
                # throughputs.append(throughput)

                add_item_to_dict_by_agent_num(
                    all_throughputs_dict,
                    agent_num,
                    throughput,
                )
                add_item_to_dict_by_agent_num(
                    all_runtime_dict,
                    agent_num,
                    runtime,
                )
                add_item_to_dict_by_agent_num(
                    all_success_dict,
                    agent_num,
                    success,
                )

            else:
                print(f"Result of {sim_dir} is missing")

        # sort_idx = np.argsort(agent_nums)
        # agent_nums = np.array(agent_nums)[sort_idx]
        # throughputs = np.array(throughputs)[sort_idx]

        # all_throughputs_dict.append(throughputs)

    # all_throughputs_dict = sorted(all_throughputs_dict.items())
    # agent_nums = [agent_num for agent_num, _ in all_throughputs_dict]
    # all_throughputs_vals = [
    #     throughputs for _, throughputs in all_throughputs_dict
    # ]

    agent_nums, all_throughputs_vals = sort_and_get_vals_from_dict(
        all_throughputs_dict)
    _, all_runtime_vals = sort_and_get_vals_from_dict(all_runtime_dict)
    _, all_success_vals = sort_and_get_vals_from_dict(all_success_dict)

    save_fig = False
    if ax is None:
        save_fig = True
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Take the average, confidence interval and standard error
    # all_throughputs = np.array(all_throughputs_vals)
    # mean_throughputs = np.mean(all_throughputs, axis=1)
    # cf_throughputs = st.t.interval(confidence=0.95,
    #                                df=len(all_throughputs) - 1,
    #                                loc=mean_throughputs,
    #                                scale=st.sem(all_throughputs, axis=1) + 1e-8)
    # sem_throughputs = st.sem(all_throughputs, axis=1)

    (
        mean_throughputs,
        cf_throughputs,
        sem_throughputs,
        mean_throughputs_success,
        sem_throughputs_success,
    ) = compute_numerical(all_throughputs_vals, all_success_vals)

    (
        mean_runtime,
        _,
        sem_runtime,
        mean_runtime_success,
        sem_runtime_success,
    ) = compute_numerical(all_runtime_vals, all_success_vals)

    all_success_vals = np.array(all_success_vals)
    success_rates = np.sum(all_success_vals, axis=1) / all_success_vals.shape[1]
    # breakpoint()

    color = get_color(map_from, algo_name, n_agents_opt)
    label = f"{map_from} + {algo_name_map[algo_name]}"
    if n_agents_opt is not None:
        label += f"({n_agents_opt} agents)"
    ax.plot(
        agent_nums,
        mean_throughputs,
        # marker=".",
        color=color,
        label=label,
        # label=f"{map_from}",
    )
    ax.fill_between(
        agent_nums,
        cf_throughputs[1],
        cf_throughputs[0],
        alpha=0.5,
        color=color,
    )

    if save_fig:
        ax.set_ylabel("Throughput", fontsize=25)
        ax.set_xlabel("Number of Agents", fontsize=25)
        ax.set_ylim(y_min, y_max)
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=15)

        ax.figure.tight_layout()
        fig.savefig(
            os.path.join(
                logdirs_plot,
                f"throughput_agentNum_{algo_name}_{map_size}_{mode}.png",
            ),
            dpi=300,
        )

    # Create numerical result
    numerical_result = {}
    numerical_result["agent_num"] = agent_nums
    numerical_result["mean_throughput"] = mean_throughputs
    numerical_result["mean_throughput_success"] = mean_throughputs_success
    numerical_result["sem_throughput"] = sem_throughputs
    numerical_result["sem_throughputs_success"] = sem_throughputs_success
    numerical_result["mean_runtime"] = mean_runtime
    numerical_result["mean_runtime_success"] = mean_runtime_success
    numerical_result["sem_runtime"] = sem_runtime
    numerical_result["sem_runtime_success"] = sem_runtime_success
    numerical_result["success_rate"] = success_rates
    numerical_result_df = pd.DataFrame(numerical_result)
    numerical_result_df.to_csv(
        os.path.join(
            logdirs_plot,
            f"numerical_{algo_name}_{map_size}_{mode}.csv",
        ))

    return agent_nums, y_min, y_max, meta


if __name__ == "__main__":
    fire.Fire(throughput_vs_n_agents)

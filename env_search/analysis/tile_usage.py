from operator import index
import os
from typing import List
import gin
import json
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import fire
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from logdir import LogDir
from pathlib import Path
from pprint import pprint
from env_search.analysis.utils import load_experiment, load_metrics
from env_search.analysis.visualize_env import visualize_kiva
from env_search.utils.logging import setup_logging
from env_search.mpl_styles.utils import mpl_style_file
from env_search.utils import (set_spines_visible, KIVA_ROBOT_BLOCK_WIDTH,
                              KIVA_WORKSTATION_BLOCK_WIDTH,
                              KIVA_ROBOT_BLOCK_HEIGHT, kiva_env_number2str,
                              kiva_env_str2number, read_in_kiva_map)

mpl.use("agg")

# set matplotlib params
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def get_tile_usage_vmax(env_h, env_w):
    if env_h * env_w == 9 * 16 or env_h * env_w == 9 * 20:
        vmax = 0.03
    elif env_h * env_w == 17 * 16 or env_h * env_w == 17 * 20:
        vmax = 0.02
    elif env_h * env_w == 33 * 36:
        vmax = 0.01
    else:
        vmax = 0.04
    return vmax


def plot_tile_usage(
    tile_usage,
    env_h,
    env_w,
    fig,
    ax_tile_use,
    ax_tile_use_cbar,
    logdir,
    filenames: List = ["tile_usage.pdf", "tile_usage.svg", "tile_usage.png"],
    dpi=300,
):
    # Plot tile usage
    tile_usage = tile_usage.reshape(env_h, env_w)

    sns.heatmap(
        tile_usage,
        square=True,
        cmap="Reds",
        ax=ax_tile_use,
        # cbar_ax=ax_tile_use_cbar,
        cbar=True,
        rasterized=False,
        annot_kws={"size": 30},
        linewidths=1,
        linecolor="black",
        xticklabels=False,
        yticklabels=False,
        vmin=0,
        vmax=get_tile_usage_vmax(env_h, env_w),
        cbar_kws={
            # "orientation": "horizontal",
            "shrink": 0.7,
        },
    )

    cbar = ax_tile_use.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    set_spines_visible(ax_tile_use)
    ax_tile_use.figure.tight_layout()

    for filename in filenames:
        fig.savefig(logdir.file(filename), dpi=dpi)


def get_figsize_qd(w_mode=True):
    # Decide figsize based on size of map
    sol_sim = gin.query_parameter("Manager.sol_size")
    if sol_sim == 108:
        if w_mode:
            figsize = (8, 8)
        else:
            figsize = (8, 12)
    elif sol_sim == 204:
        figsize = (8, 16)
    elif sol_sim == 1056:
        figsize = (8, 16)
    else:
        figsize = (8, 8)

    return figsize


def get_figsize_sim(env_np):
    # Decide figsize based on size of map
    env_h, env_w = env_np.shape
    if env_h * env_w == 9 * 16 or env_h * env_w == 9 * 20:
        figsize = (8, 8)
    elif env_h * env_w == 17 * 16 or env_h * env_w == 17 * 20:
        figsize = (8, 16)
    elif env_h * env_w == 33 * 36:
        figsize = (8, 16)
    else:
        figsize = (8, 8)

    return figsize


def tile_usage_heatmap_from_single_run(logdir: str, dpi=300):
    """
    Plot tile usage with map layout from a single run of warehouse simulation.
    """
    # Read in map
    map_filepath = os.path.join(logdir, "map.json")
    map, map_name = read_in_kiva_map(map_filepath)
    env_np = kiva_env_str2number(map)

    # Create plot
    grid_kws = {"height_ratios": (0.475, 0.475, 0.05)}
    fig, (ax_map, ax_tile_use,
          ax_tile_use_cbar) = plt.subplots(3,
                                           1,
                                           figsize=get_figsize_sim(env_np),
                                           gridspec_kw=grid_kws)

    # Plot map
    visualize_kiva(env_np, ax=ax_map, dpi=300)

    # Read in result and plot tile usage
    results_dir = os.path.join(logdir, "results")
    for sim_dir in tqdm(os.listdir(results_dir)):
        sim_dir_comp = os.path.join(results_dir, sim_dir)
        result_filepath = os.path.join(sim_dir_comp, "result.json")
        with open(result_filepath, "r") as f:
            result_json = json.load(f)
        tile_usage = result_json["tile_usage"]
        plot_tile_usage(
            np.array(tile_usage),
            env_np.shape[0],
            env_np.shape[1],
            fig,
            ax_tile_use,
            ax_tile_use_cbar,
            LogDir(map_name, custom_dir=sim_dir_comp),
            filenames=[
                f"tile_usage_{map_name}.pdf",
                f"tile_usage_{map_name}.svg",
                f"tile_usage_{map_name}.png",
            ],
            dpi=dpi,
        )


def tile_usage_heatmap_from_qd(
    logdir: str,
    gen: int = None,
    index_0: int = None,
    index_1: int = None,
    dpi: int = 300,
    mode: str = None,
):
    """
    Plot tile usage with map layout from a QD experiment.
    """
    logdir = load_experiment(logdir)
    gen = load_metrics(logdir).total_itrs if gen is None else gen
    df = pd.read_pickle(logdir.file(f"archive/archive_{gen}.pkl"))
    global_opt_env = None

    if index_0 is not None and index_1 is not None:
        to_plots = df[(df["index_0"] == index_0) & (df["index_1"] == index_1)]
        if to_plots.empty:
            raise ValueError("Specified index has no solution")
    elif mode == "extreme":
        # Plot the "extreme" points in the archive
        # Choose the largest measure1 or measure2. If tie, choose the one with
        # larger objective.
        # df.loc[df["index_0"]==df["index_0"].max()]["objective"].idxmax()

        index_0_max = df.loc[df["index_0"] ==
                             df["index_0"].max()]["objective"].idxmax()
        index_0_min = df.loc[df["index_0"] ==
                             df["index_0"].min()]["objective"].idxmax()
        index_1_max = df.loc[df["index_1"] ==
                             df["index_1"].max()]["objective"].idxmax()
        index_1_min = df.loc[df["index_1"] ==
                             df["index_1"].min()]["objective"].idxmax()

        # Add global optimal env
        global_opt = df["objective"].idxmax()
        global_opt_env = df.iloc[global_opt]["metadata"]["warehouse_metadata"][
            "map_str"]
        to_plots = df.iloc[[
            index_0_max,
            index_0_min,
            index_1_max,
            index_1_min,
            global_opt,
        ]]
    elif mode == "extreme-3D":
        # In 3D case, we fix the third dimension and plot the "extreme" points
        # in the archive in first/second dimensions
        partial_df = df[df["behavior_2"] == 20]
        index_0_max = partial_df["index_0"].idxmax()
        index_0_min = partial_df["index_0"].idxmin()
        index_1_max = partial_df["index_1"].idxmax()
        index_1_min = partial_df["index_1"].idxmin()

        # Add global optimal env
        global_opt = partial_df["objective"].idxmax()
        global_opt_env = partial_df.iloc[global_opt]["metadata"][
            "warehouse_metadata"]["map_str"]
        to_plots = partial_df.loc[[
            index_0_max,
            index_0_min,
            index_1_max,
            index_1_min,
            global_opt,
        ]]
    elif mode == "compare_human":
        selected_inds = df[(df["behavior_1"] == 20)]
        to_plots = []
        if not selected_inds.empty:
            curr_opt_idx_20 = selected_inds["objective"].idxmax()
            to_plots.append(df.iloc[curr_opt_idx_20])
        else:
            print("No map with 20 shelves in the archive!")
        selected_inds = df[(df["behavior_1"] == 24)]
        if not selected_inds.empty:
            curr_opt_idx_24 = selected_inds["objective"].idxmax()
            to_plots.append(df.iloc[curr_opt_idx_24])
        else:
            print("No map with 24 shelves in the archive!")

        to_plots = pd.DataFrame(to_plots)

    if global_opt_env is not None:
        print("Global optima: ")
        print("\n".join(global_opt_env))
        print()

    with mpl_style_file("tile_usage_heatmap.mplstyle") as f:
        with plt.style.context(f):
            for _, to_plot in to_plots.iterrows():
                index_0 = to_plot["index_0"]
                index_1 = to_plot["index_1"]
                obj = to_plot["objective"]
                metadata = to_plot["metadata"]["warehouse_metadata"]
                throughput = np.mean(metadata["throughput"])
                print(
                    f"Index ({index_0}, {index_1}): objective = {obj}, throughput = {throughput}"
                )
                w_mode = gin.query_parameter("WarehouseManager.w_mode")

                grid_kws = {"height_ratios": (0.475, 0.475, 0.05)}
                fig, (ax_map, ax_tile_use, ax_tile_use_cbar) = plt.subplots(
                    3,
                    1,
                    figsize=get_figsize_qd(w_mode),
                    gridspec_kw=grid_kws,
                )

                # Plot repaired env
                repaired_env_str = metadata["map_str"]
                print("\n".join(repaired_env_str))
                print()
                repaired_env = kiva_env_str2number(repaired_env_str)
                visualize_kiva(repaired_env, ax=ax_map, dpi=300)

                tile_usage = np.array(metadata["tile_usage"])
                env_h = gin.query_parameter("WarehouseManager.lvl_height")
                env_w = gin.query_parameter("WarehouseManager.lvl_width")
                ADDITION_BLOCK_WIDTH = KIVA_WORKSTATION_BLOCK_WIDTH if w_mode \
                                       else KIVA_ROBOT_BLOCK_WIDTH
                ADDITION_BLOCK_HEIGHT = 0 if w_mode else KIVA_ROBOT_BLOCK_HEIGHT
                env_w += 2 * ADDITION_BLOCK_WIDTH
                env_h += 2 * ADDITION_BLOCK_HEIGHT
                if len(tile_usage.shape) == 2:
                    tile_usage = tile_usage[np.newaxis, ...]

                # mkdir for tileusage
                tile_usage_dir = logdir.dir("tile_usages")

                for i in range(tile_usage.shape[0]):
                    curr_tile_usage = tile_usage[i]
                    plot_tile_usage(
                        curr_tile_usage,
                        env_h,
                        env_w,
                        fig,
                        ax_tile_use,
                        ax_tile_use_cbar,
                        logdir,
                        filenames=[
                            f"tile_usage/{index_0}_{index_1}-{i}.pdf",
                            f"tile_usage/{index_0}_{index_1}-{i}.svg",
                            f"tile_usage/{index_0}_{index_1}-{i}.png",
                        ],
                        dpi=dpi,
                    )

                plt.close('all')


def main(
    logdir: str,
    logdir_type: str = "qd",  # "qd" or "sim"
    gen: int = None,
    index_0: int = None,
    index_1: int = None,
    dpi: int = 300,
    mode: str = None,
):
    if logdir_type == "qd":
        tile_usage_heatmap_from_qd(
            logdir=logdir,
            gen=gen,
            index_0=index_0,
            index_1=index_1,
            dpi=dpi,
            mode=mode,
        )
    elif logdir_type == "sim":
        tile_usage_heatmap_from_single_run(logdir, dpi=dpi)


if __name__ == "__main__":
    fire.Fire(main)
import os
import shutil

import torch
import fire
import gin
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

from loguru import logger

from env_search.device import DEVICE
from env_search.warehouse.emulation_model.emulation_model import WarehouseEmulationModel
from env_search.analysis.utils import load_archive_gen, load_experiment, load_metrics
from env_search.utils import kiva_env_str2number, flip_tiles
from env_search.mpl_styles.utils import mpl_style_file
from env_search.analysis.visualize_env import visualize_kiva
from env_search.analysis.tile_usage import get_figsize_qd, plot_tile_usage


def main(
    logdir: str,
    gen: int = None,
    offline_logdir_data: str = None,
    w_mode: bool = True,
):
    logdir = load_experiment(logdir)
    reload_pth = logdir.pfile("reload_em.pth")
    reload_pkl = logdir.pfile("reload_em.pkl")

    # Read in master seed
    with open(logdir.pfile("seed"), "r") as f:
        seed = int(f.readline())

    # Load trained model
    model = WarehouseEmulationModel(seed=seed + 420)
    model.load(reload_pkl, reload_pth, map_location="cpu")

    breakpoint()

    # The model is trained offline, need to read test map from logdir that
    # contains the training data.
    if offline_logdir_data:
        reload_em_pkl = os.path.join(offline_logdir_data, "reload_em.pkl")
        with open(reload_em_pkl, "rb") as f:
            data = pkl.load(f)
            dataset = data["dataset"]
        env_unrepaired = dataset.levels[0]
        env_repaired = dataset.repaired_maps[0]
        occupancy = dataset.occupancys[0]
        true_obj = dataset.objectives[0]
        true_measure1, true_measure2 = dataset.measures[0]

        if w_mode:
            env_repaired = flip_tiles(
                env_repaired,
                'r',
                'w',
            )

    else:
        # Load one of the training maps from the archive
        gen = load_metrics(logdir).total_itrs if gen is None else gen
        df = pd.read_pickle(logdir.file(f"archive/archive_{gen}.pkl"))

        global_opt = df["objective"].idxmax()
        env_unrepaired = df.iloc[global_opt]["metadata"]["warehouse_metadata"][
            "map_int_unrepaired"]
        env_repaired = df.iloc[global_opt]["metadata"]["warehouse_metadata"][
            "map_int"]
        occupancy = df.iloc[global_opt]["metadata"]["warehouse_metadata"][
            "tile_usage"]
        occupancy = np.mean(occupancy, axis=0)

        # Process the map
        # Maps in archive contains 'w' while maps in dataset are already
        # processed
        if w_mode:
            env_unrepaired = flip_tiles(
                env_unrepaired,
                'w',
                'r',
            )

        # Get real value
        true_obj = df.iloc[global_opt]["objective"]
        true_measure1 = df.iloc[global_opt]["behavior_0"]
        true_measure2 = df.iloc[global_opt]["behavior_1"]

    env_unrepaired = env_unrepaired[np.newaxis, ...]

    # Compute the repaired map and tile usage map
    with torch.no_grad():
        pred_repaired_map, pred_occupancy = \
            model.pre_network.int_to_logits(
                torch.as_tensor(env_unrepaired, device=DEVICE))
        pred_occupancy = torch.softmax(torch.flatten(pred_occupancy), dim=0)
        pred_occupancy = torch.reshape(pred_occupancy, occupancy.shape)
    pred_repaired_map = pred_repaired_map.numpy()
    pred_repaired_map = np.argmax(pred_repaired_map, axis=1)[0]
    if w_mode:
        pred_repaired_map = flip_tiles(
            pred_repaired_map,
            'r',
            'w',
        )

    # Create plot
    grid_kws = {"height_ratios": (0.475, 0.475, 0.05)}
    pred_fig, (ax_pred_map, ax_pred_tile_use,
          ax_pred_tile_use_cbar) = plt.subplots(3,
                                           1,
                                           figsize=get_figsize_qd(),
                                           gridspec_kw=grid_kws)
    fig, (ax_map, ax_tile_use,
        ax_tile_use_cbar) = plt.subplots(3,
                                        1,
                                        figsize=get_figsize_qd(),
                                        gridspec_kw=grid_kws)
    # Plot and save to disk
    validate_mdl_dir = logdir.dir("validate_mdl")
    if os.path.exists(validate_mdl_dir):
        shutil.rmtree(validate_mdl_dir)
    os.mkdir(validate_mdl_dir)

    # Plot unrepaired map
    visualize_kiva(
        env_unrepaired[0],
        filenames = ["unrepaired.png"],
        store_dir = validate_mdl_dir,
    )

    # Plot ground-truth repaired map with pred tile usage
    visualize_kiva(
        env_repaired,
        ax=ax_map,
    )
    plot_tile_usage(
        occupancy,
        occupancy.shape[0],
        occupancy.shape[1],
        fig,
        ax_tile_use,
        ax_tile_use_cbar,
        logdir,
        filenames=[
            f"validate_mdl/tile_usage.png",
        ],
        dpi=300,
    )

    # Plot pred repaired map with pred tile usage
    visualize_kiva(
        pred_repaired_map,
        ax=ax_pred_map,
    )
    plot_tile_usage(
        pred_occupancy,
        pred_occupancy.shape[0],
        pred_occupancy.shape[1],
        pred_fig,
        ax_pred_tile_use,
        ax_pred_tile_use_cbar,
        logdir,
        filenames=[
            f"validate_mdl/pred_tile_usage.png",
        ],
        dpi=300,
    )

    # Compute the final obj and measures
    obj, measures = model.predict(env_unrepaired)
    print(f"Predicted obj: {obj[0]}")
    print(f"Predicted measures: [{measures[0][0]}, {measures[0][1]}]")

    print(f"True obj: {true_obj}")
    print(f"True measures: [{true_measure1}, {true_measure2}]")

if __name__ == '__main__':
    fire.Fire(main)
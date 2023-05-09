"""Code for plotting the correlation graph between the predicted and true
measures.

After running the experiments, copy `reload_em.pkl` from the logging directory
of one run of each surrogate assisted algorithm.

Specify the logging directory of the other run in a JSON file with the following
format:
```
{
    "Maze": {"DSAGE": ..., "DSAGE Basic": ...},
    "Mario": {"DSAGE": ..., "DSAGE Basic": ...}
}
```

Run the script with
`python -m src.analysis.prediction_plots --logdir_json=<JSON file>`
to generate the plots.
"""
import json
import pickle

import fire
import gin
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.analysis.figures import COLORBLIND_REORDERED, legend_info
from src.main import setup_logdir
from src.mario.emulation_model.emulation_model import MarioEmulationModel
from src.maze.emulation_model.emulation_model import MazeEmulationModel
from src.mpl_styles.utils import mpl_style_file


def main(logdir_json):
    domain_list = ["Maze", "Mario"]
    alg_list = ["DSAGE", "DSAGE Basic"]
    line_types = ["Ground-truth", "Prediction"]
    measure_names = {
        "Maze": ["Number of wall cells", "Mean agent path length"],
        "Mario": ["Sky tiles", "Number of jumps"]
    }
    ems = {"Maze": MazeEmulationModel, "Mario": MarioEmulationModel}
    config_files = {
        "Maze": {
            "DSAGE":
                "config/maze/dsage.gin",
            "DSAGE Basic":
                "config/maze/dsage_basic.gin",
        },
        "Mario": {
            "DSAGE": "config/mario/dsage.gin",
            "DSAGE Basic": "config/mario/dsage_basic.gin",
        }
    }

    with open(logdir_json) as f:
        logdirs = json.load(f)

    tols = {
        "Maze": {
            "Number of wall cells": 0.5,
            "Mean agent path length": 2,
        },
        "Mario": {
            "Sky tiles": 0.5,
            "Number of jumps": 0.5,
        }
    }

    plot_data = {
        "True Measure Cell": [],
        "Pred. Measure Cell": [],
        "Line Type": [],
        "Measure": [],
        "Algorithm": [],
    }

    for domain in domain_list:
        data_files = [
            f"{domain.lower()}_base_reload_em.pkl",
            f"{domain.lower()}_occ_reload_em.pkl",
            f"{domain.lower()}_down_reload_em.pkl",
            f"{domain.lower()}_reload_em.pkl"
        ]

        levels, true_objs, true_measures = [], [], []
        for data_file in data_files:
            with open(data_file, "rb") as f:
                data = pickle.load(f)
            dataset = data["dataset"]
            levels.append(np.array(dataset.levels))
            true_objs.append(np.array(dataset.objectives))
            true_measures.append(np.array(dataset.measures))
            print(f"{len(dataset.levels)} levels in {data_file}")

        levels = np.concatenate(levels)
        true_objs = np.concatenate(true_objs)
        true_measures = np.concatenate(true_measures)
        print(f"{len(levels)} levels in total")

        for alg in alg_list:
            print(f"Domain: {domain}; Alg: {alg}")
            gin.clear_config(clear_constants=True)
            gin.parse_config_file(config_files[domain][alg])
            logdir = setup_logdir(42, None, logdirs[domain][alg])
            em = ems[domain](seed=42)
            em.load(logdir.pfile("reload_em.pkl"),
                    logdir.pfile("reload_em.pth"))

            pred_objs, pred_measures = [], []
            for i in range(0, len(levels), batch_size := 1000):
                pred_obj, pred_measure = em.predict(levels[i:i + batch_size])
                pred_objs.append(pred_obj)
                pred_measures.append(pred_measure)

            pred_objs = np.concatenate(pred_objs)
            pred_measures = np.concatenate(pred_measures)

            for i, mname in enumerate(measure_names[domain]):
                pred_cells = pred_measures[:, i] // (tols[domain][mname] * 2)
                true_cells = true_measures[:, i] // (tols[domain][mname] * 2)

                sort_idx = np.argsort(true_cells)
                xs = true_cells[sort_idx]
                ys = pred_cells[sort_idx]
                df = pd.DataFrame({"true": xs, "pred": ys})
                means = df.groupby(["true"]).mean()

                x_means = means.index.to_numpy()
                y_means = means["pred"].to_numpy()
                data_len = len(x_means)

                for ltype in line_types:
                    if ltype == "Ground-truth":
                        y_data = x_means
                    else:
                        y_data = y_means

                    plot_data["True Measure Cell"].append(x_means)
                    plot_data["Pred. Measure Cell"].append(y_data)
                    plot_data["Line Type"].append(np.full(data_len, ltype))
                    plot_data["Measure"].append(np.full(data_len, mname))
                    plot_data["Algorithm"].append(np.full(data_len, alg))

    # Flatten everything so that Seaborn understands it.
    for d in plot_data:
        plot_data[d] = np.concatenate(plot_data[d])

    with mpl_style_file("comparison.mplstyle") as f:
        with plt.style.context(f):
            colors = COLORBLIND_REORDERED

            palette = dict(zip(line_types, colors[:2]))
            markers = dict(zip(line_types, "ov"))
            grid = sns.relplot(
                data=plot_data,
                x="True Measure Cell",
                y="Pred. Measure Cell",
                hue="Line Type",
                style="Line Type",
                col="Measure",
                row="Algorithm",
                kind="line",
                markers=markers,
                markevery=(0.3, 0.4),
                markersize=10,
                dashes=False,
                height=2.5,
                aspect=1,
                facet_kws={
                    "sharey": False,
                    "sharex": False
                },
                palette=palette,
                legend=False,
            )

            grid.set_titles("{col_name}", size=16, pad=20)
            for ax in grid.axes[1:].ravel():
                ax.set_title("")

            for (row_val, col_val), ax in grid.axes_dict.items():
                print("RC", row_val, col_val)
                # Move the ticks and grid lines below the plotted lines.
                ax.set_axisbelow(True)

                # Turn off both labels.
                # ax.set_xlabel("")
                # ax.set_ylabel("")

                if col_val == "Number of wall cells":
                    ax.set_xticks([0, 128, 256])
                    # ax.set_xticklabels([0, "5k", "10k"])
                    ax.set_xlim(0, 256)
                    ax.set_yticks([0, 128, 256])
                    # ax.set_yticklabels([0, "2.5k", "5k"])
                    ax.set_ylim(0, 256)
                elif col_val == "Mean agent path length":
                    ax.set_xticks([0, 81, 162])
                    ax.set_xlim(0, 162)
                    ax.set_yticks([0, 81, 162])
                    ax.set_ylim(0, 162)
                elif col_val == "Sky tiles":
                    ax.set_xticks([0, 75, 150])
                    ax.set_xlim(0, 150)
                    ax.set_yticks([0, 75, 150])
                    ax.set_ylim(0, 150)
                elif col_val == "Number of jumps":
                    ax.set_xticks([0, 30, 60])
                    ax.set_xlim(0, 60)
                    ax.set_yticks([0, 30, 60])
                    ax.set_ylim(0, 60)

            grid.fig.legend(
                *legend_info(line_types, palette, markers),
                bbox_to_anchor=[0.5, 1.0],
                loc="upper center",
                ncol=2,
                # # Slightly smaller than the main font size.
                # fontsize=14.5,
                # Smaller than default of 2.0 to keep legend columns close.
                columnspacing=1.0,
                # Padding between handle icon and text.
                handletextpad=0.4,
            )

            # Row labels
            grid.fig.text(
                x=0,
                y=0.67,
                verticalalignment="center",
                s="DSAGE",  # this is the text in the xlabel
                size=16,
                rotation=90)

            grid.fig.text(
                x=0,
                y=0.28,
                verticalalignment="center",
                s="DSAGE Basic",  # this is the text in the xlabel
                size=16,
                rotation=90)

            fig_width, fig_height = grid.fig.get_size_inches()
            legend_height = 0.4
            grid.fig.set_size_inches(fig_width, fig_height + legend_height)

            # Save the figure.
            grid.fig.tight_layout(rect=(0.02, 0, 1, fig_height /
                                        (fig_height + legend_height)))

            for ext in ["pdf", "png"]:
                grid.fig.savefig(f"temp/pred_corrs.{ext}")


if __name__ == '__main__':
    fire.Fire(main)

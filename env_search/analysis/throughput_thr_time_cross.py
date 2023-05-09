import os

import fire
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from env_search.analysis.throughput_thr_time import throughput_thr_time

mpl.use("agg")

# set matplotlib params
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})



def throughput_thr_time_cross(
    all_logdirs_plot,
    ax=None,
    add_legend=False,
    save_fig=True,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i, logdirs_plot_dir in enumerate(os.listdir(all_logdirs_plot)):
        logdirs_plot = os.path.join(all_logdirs_plot, logdirs_plot_dir)
        if not os.path.isdir(logdirs_plot):
            continue
        timesteps, y_min, y_max, meta = throughput_thr_time(logdirs_plot, ax=ax)

    ax.set_ylabel("Number of Finished Tasks", fontsize=45)
    ax.set_xlabel("Timestep", fontsize=45)
    ax.set_xlim(0, timesteps[-1] + 5)
    ax.set_ylim(y_min, y_max)
    # ax.grid()
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.tick_params(axis='both', which='minor', labelsize=15)
    range_t = [0, timesteps[-1] + 5]
    range_y = [y_min, y_max]
    ax.set_xticks([range_t[0], np.mean([range_t]), range_t[1]])
    ax.set_yticks([range_y[0], np.mean([range_y]), range_y[1]])

    ax.set_xticklabels(
        [range_t[0], np.mean([range_t], dtype=int), range_t[1]],
        rotation=0,
        fontsize=35)
    ax.set_yticklabels([range_y[0], np.mean([range_y]), range_y[1]],
                       fontsize=35)

    # Legend
    if add_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            loc="lower left",
            ncol=2,
            fontsize=22,
            mode="expand",
            bbox_to_anchor=(0, 1.02, 1, 0.2),  # for ncols=2
            # borderaxespad=0,)
        )

    if save_fig:
        mode = meta["mode"]
        map_size = meta["map_size"]
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                all_logdirs_plot,
                f"thr_time_{mode}_{map_size}.pdf",
            ),
            dpi=300,
        )

        fig.savefig(
            os.path.join(
                all_logdirs_plot,
                f"thr_time_{mode}_{map_size}.png",
            ),
            dpi=300,
        )


if __name__ == "__main__":
    fire.Fire(throughput_thr_time_cross)
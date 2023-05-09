"""Visualizes throughput vs. n_agents from multiple experiments.

Usage:
    python env_search/analysis/throughput_vs_n_agents_cross.py --all_logdirs_plot <all_logdirs_plot>
"""
import os
import fire
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from env_search.analysis.throughput_vs_n_agents import throughput_vs_n_agents

mpl.use("agg")

# set matplotlib params
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def throughput_vs_n_agents_cross(
    all_logdirs_plot,
    ax=None,
    add_legend=True,
    save_fig=True,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i, logdirs_plot_dir in enumerate(sorted(os.listdir(all_logdirs_plot))):
        logdirs_plot = os.path.join(all_logdirs_plot, logdirs_plot_dir)
        if not os.path.isdir(logdirs_plot):
            continue
        agent_nums, y_min, y_max, meta = throughput_vs_n_agents(logdirs_plot,
                                                                ax=ax)

    # ax.set_xlim(agent_nums[0], agent_nums[-1])
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Throughput", fontsize=45)
    ax.set_xlabel("Number of Agents", fontsize=45)
    # ax.grid()
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.tick_params(axis='both', which='minor', labelsize=15)

    # Add a vertical line at the number of agents used to optimize the layout
    map_size = meta["map_size"]
    mode = meta["mode"]

    n_agent_vertical = None
    if map_size == "small":
        if mode == "w":
            n_agent_vertical = 60
        elif mode == "r":
            n_agent_vertical = 88
    elif map_size == "medium":
        n_agent_vertical = 90
    elif map_size == "large":
        n_agent_vertical = 200

    # Draw vertical line
    # ax.axvline(x=n_agent_vertical, color="black", linewidth=4)

    range_x = [agent_nums[0], agent_nums[-1]]
    range_y = [y_min, y_max]
    if map_size != "large":
        ax.set_xticks([
            range_x[0],
            np.mean([range_x]),
            n_agent_vertical,
            range_x[1],
        ])
        ax.set_xticklabels(
        [
            range_x[0],
            np.mean([range_x], dtype=int),
            n_agent_vertical,
            range_x[1],
        ],
        rotation=0,
        fontsize=35,
    )
    elif map_size == "large":
        ax.set_xticks([
            range_x[0],
            n_agent_vertical,
            range_x[1],
        ])
        ax.set_xticklabels(
        [
            range_x[0],
            n_agent_vertical,
            range_x[1],
        ],
        rotation=0,
        fontsize=35,
    )
    y_mid = np.mean([range_y])
    if y_mid.is_integer():
        y_mid = int(y_mid)
    ax.set_yticks([range_y[0], y_mid, range_y[1]])
    ax.set_yticklabels(
        [range_y[0], y_mid, range_y[1]],
        fontsize=35,
    )

    # Legend
    if add_legend:
        handles, labels = ax.get_legend_handles_labels()
        # ax.legend(
        #     handles,
        #     labels,
        #     loc="lower left",
        #     ncol=2,
        #     fontsize=35,
        #     mode="expand",
        #     bbox_to_anchor=(0, 1.02, 1, 0.2),  # for ncols=2
        #     # borderaxespad=0,)
        # )


        # For front fig
        # order = [1, 0]
        # ax.legend(
        #     [handles[idx] for idx in order],
        #     [labels[idx] for idx in order],
        #     # loc="lower left",
        #     ncol=1,
        #     fontsize=35,
        #     frameon=False,
        #     # mode="expand",
        #     # bbox_to_anchor=(0, 1.02, 1, 0.2),  # for ncols=2
        #     # borderaxespad=0,)
        # )

        # For r-mode less agents
        ax.legend(
            handles,
            labels,
            # loc="lower left",
            ncol=1,
            fontsize=35,
            borderaxespad=0,
            bbox_to_anchor=(1.04, 1),  # for ncols=2
        )


    if save_fig:
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                all_logdirs_plot,
                f"agent_num_{mode}_{map_size}.pdf",
            ),
            dpi=300,
            bbox_inches='tight',
        )

        fig.savefig(
            os.path.join(
                all_logdirs_plot,
                f"agent_num_{mode}_{map_size}.png",
            ),
            dpi=300,
            bbox_inches='tight',
        )


if __name__ == "__main__":
    fire.Fire(throughput_vs_n_agents_cross)
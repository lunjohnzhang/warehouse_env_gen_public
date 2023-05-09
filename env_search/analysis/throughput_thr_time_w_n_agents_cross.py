import os
import fire

import matplotlib as mpl
import matplotlib.pyplot as plt

from env_search.analysis.throughput_vs_n_agents_cross import throughput_vs_n_agents_cross

from env_search.analysis.throughput_thr_time_cross import throughput_thr_time_cross

mpl.use("agg")

# set matplotlib params
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def throughput_thr_time_w_n_agents_cross(
    all_thr_time_logdirs_plot,
    all_n_agents_logdirs_plot,
):
    fig, (ax_n_agents, ax_thr_time) = plt.subplots(1, 2, figsize=(22, 10))
    throughput_thr_time_cross(
        all_thr_time_logdirs_plot,
        ax=ax_thr_time,
        add_legend=False,
        save_fig=False,
    )
    throughput_vs_n_agents_cross(
        all_n_agents_logdirs_plot,
        ax=ax_n_agents,
        add_legend=False,
        save_fig=False,
    )

    handles, labels = ax_n_agents.get_legend_handles_labels()
    if len(labels) == 6:
        order = [0, 3, 2, 5, 1, 4]
    else:
        order = [i for i in range(len(labels))]
    fig.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc="upper center",
        ncol=2,
        fontsize=35,
        # mode="expand",
        bbox_to_anchor=(0.5, 1.10),  # for ncols=2
    )
    # fig.tight_layout()
    fig.savefig(
        os.path.join(
            "logs",
            f"throughput_thr_time_w_n_agents_cross.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        os.path.join(
            "logs",
            f"throughput_thr_time_w_n_agents_cross.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == '__main__':
    fire.Fire(throughput_thr_time_w_n_agents_cross)
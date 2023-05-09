"""Visualizes heatmaps from an experiment.

Usage:
    python -m env_search.analysis.heatmap -l LOGDIR
"""
import os
import shutil

import fire
import gin
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ribs.visualize
from env_search.archives import GridArchive
from loguru import logger

from env_search.analysis.utils import load_archive_gen, load_experiment, load_metrics, load_surrogate_archive, grid_archive_heatmap
from env_search.mpl_styles.utils import mpl_style_file

mpl.use("agg")

# set matplotlib params
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MEASURE_NAMES = {
    "n_shelf": "Number of shelves",
    "tile_usage_std": "Tile usage std",
    "num_wait_mean": "Avg number of wait",
    "all_task_len_mean": "Mean task length",
    "n_shelf_components": "Number of shelf components",
}


def plot_generation_on_axis(
    ax,
    surr_ax,
    downsample_ax,
    mode,
    logdir,
    gen,
    plot_kwargs,
):
    # pylint: disable = unused-argument
    archive = load_archive_gen(logdir, gen)
    grid_archive_heatmap(
        archive,
        ax,
        transpose_bcs=True,
        cbar_ticklabel_fontsize=20,
        **plot_kwargs,
    )

    is_em = gin.query_parameter("Manager.is_em")
    if is_em and os.path.isdir(logdir.dir("surrogate_archive")):
        surr_archive, downsample_archive = load_surrogate_archive(logdir, -1)
        grid_archive_heatmap(
            surr_archive,
            surr_ax,
            transpose_bcs=True,
            cbar_ticklabel_fontsize=20,
            **plot_kwargs,
        )
        if downsample_archive:
            grid_archive_heatmap(
                downsample_archive,
                downsample_ax,
                transpose_bcs=True,
                cbar_ticklabel_fontsize=20,
                **plot_kwargs,
            )


def post_process_figure(ax, fig, kiva, heatmap_only):
    if kiva:
        if heatmap_only:
            ax.tick_params(
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )

        else:
            sol_sim = gin.query_parameter("Manager.sol_size")

            measure_names = gin.query_parameter("WarehouseConfig.measure_names")
            ax.set_ylabel(MEASURE_NAMES[measure_names[0]], fontsize=26)
            ax.set_xlabel(MEASURE_NAMES[measure_names[1]], fontsize=26)

            range_y, range_x = gin.query_parameter("GridArchive.ranges")
            if sol_sim == 108:
                ax.set_xticks([range_x[0], range_x[1]])
                ax.set_yticks([range_y[0], range_y[1]])

                ax.set_xticklabels([range_x[0], range_x[1]],
                                   rotation=0,
                                   fontsize=20)
                ax.set_yticklabels([range_y[0], range_y[1]], fontsize=20)

            else:
                range_x_mid = np.mean(range_x)
                if range_x_mid.is_integer():
                    range_x_mid = int(range_x_mid)
                range_y_mid = np.mean(range_y)
                if range_y_mid.is_integer():
                    range_y_mid = int(range_y_mid)
                ax.set_xticks([range_x[0], range_x_mid, range_x[1]])
                ax.set_yticks([range_y[0], range_y_mid, range_y[1]])
                ax.set_xticklabels(
                    [range_x[0], range_x_mid, range_x[1]],
                    rotation=0,
                    fontsize=20)
                ax.set_yticklabels(
                    [range_y[0], range_y_mid, range_y[1]],
                    fontsize=20)

        fig.tight_layout()


def plot_generation(
    mode,
    logdir,
    gen,
    plot_kwargs,
    filenames,
    kiva,
    heatmap_only,
):
    with mpl_style_file("heatmap.mplstyle") as f:
        with plt.style.context(f):
            if kiva:
                figsize = (6.5, 5.5)

            # Figure should be created inside of the style context so that all
            # settings are handled properly, e.g. setting fonts for axis labels.
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            surr_fig, surr_ax = plt.subplots(1, 1, figsize=figsize)
            downsample_fig, downsample_ax = plt.subplots(1, 1, figsize=figsize)
            dim_x, dim_y = gin.query_parameter("GridArchive.dims")
            ax.set_box_aspect(dim_x / dim_y)
            surr_ax.set_box_aspect(dim_x / dim_y)
            downsample_ax.set_box_aspect(dim_x / dim_y)
            plot_generation_on_axis(ax, surr_ax, downsample_ax, mode, logdir,
                                    gen, plot_kwargs)
            post_process_figure(ax, fig, kiva, heatmap_only)
            post_process_figure(surr_ax, fig, kiva, heatmap_only)
            post_process_figure(downsample_ax, fig, kiva, heatmap_only)
            for filename in filenames:
                # Videos should use lower DPI because they have a lot of images.
                fig.savefig(
                    logdir.file(filename),
                    dpi="figure" if mode == "video" else 300,
                    bbox_inches='tight',
                )
                surr_fig.savefig(
                    logdir.file("surr_" + filename),
                    dpi="figure" if mode == "video" else 300,
                    bbox_inches='tight',
                )
                downsample_fig.savefig(
                    logdir.file("downsample_" + filename),
                    dpi="figure" if mode == "video" else 300,
                    bbox_inches='tight',
                )
                logger.info(f"Plotted {filename}")

    plt.close(fig)


def plot_generation_3D(mode, logdir, gen, plot_kwargs, filenames, kiva):
    with mpl_style_file("heatmap.mplstyle") as f:
        # Get number of third dimension
        dim_x, dim_y, dim_z = gin.query_parameter("GridArchive.dims")
        range_x, range_y, range_z = gin.query_parameter("GridArchive.ranges")
        with plt.style.context(f):
            if kiva:
                figsize = (6 * dim_z, 5.5)
                height_ratios = ([0.03, 0.87, 0.08])
                measure_names = gin.query_parameter(
                    "WarehouseConfig.measure_names")
            fig = plt.figure(figsize=figsize)
            spec = fig.add_gridspec(ncols=dim_z,
                                    nrows=3,
                                    hspace=0.0,
                                    height_ratios=height_ratios)
            all_ax = np.array(
                [fig.add_subplot(spec[1, i]) for i in range(dim_z)],
                dtype=object)

            # Loop through the third dimension and plot the first/second
            # dimension
            archive = load_archive_gen(logdir, gen)
            all_data = archive.as_pandas()
            all_z_vals = archive.boundaries[-1]
            for idx, z_val in enumerate(all_z_vals[:-1]):
                curr_data = all_data[
                    (all_data["behavior_2"] >= z_val)
                    & (all_data["behavior_2"] < all_z_vals[idx + 1])]
                # Create fake 2D archive that only has curr_data
                archive_type = str(gin.query_parameter("Manager.archive_type"))
                if archive_type == "@GridArchive":
                    # Same construction as in Manager.
                    # pylint: disable = no-value-for-parameter
                    partial_archive = GridArchive([dim_x, dim_y],
                                                  [range_x, range_y],
                                                  seed=42,
                                                  dtype=np.float32,
                                                  record_history=False)
                else:
                    raise TypeError(
                        f"Cannot handle archive type {archive_type}")
                partial_archive.initialize(0)  # No solutions.
                # Add obj and bcs to the partial archive
                for _, row in curr_data.iterrows():
                    # No solution and metadata, only obj and bcs of first 2 dims
                    partial_archive.add(
                        [], row["objective"],
                        np.array([row["behavior_0"], row["behavior_1"]]), None)

                # Plot current archive
                ax = all_ax[idx]
                ribs.visualize.grid_archive_heatmap(partial_archive, ax,
                                                    **plot_kwargs)
                ax.set_box_aspect(dim_y / dim_x)

                ax.set_ylabel(MEASURE_NAMES[measure_names[1]], fontsize=25)
                ax.set_xlabel(MEASURE_NAMES[measure_names[0]], fontsize=25)

                ax.set_xticks([range_x[0], np.mean([range_x]), range_x[1]])
                ax.set_yticks([range_y[0], np.mean([range_y]), range_y[1]])

                ax.set_xticklabels(
                    [range_x[0], np.mean([range_x]), range_x[1]],
                    rotation=0,
                    fontsize="x-large")
                ax.set_yticklabels(
                    [range_y[0], np.mean([range_y]), range_y[1]],
                    fontsize="x-large")
                z_measure_name = MEASURE_NAMES[measure_names[2]]
                if z_measure_name == "n_shelf":
                    z_val = int(z_val)
                ax.set_title(f"{z_measure_name} = {z_val}")

            fig.tight_layout(pad=2)

            for filename in filenames:
                # Videos should use lower DPI because they have a lot of images.
                fig.savefig(logdir.file(filename),
                            dpi="figure" if mode == "video" else 300)
                logger.info(f"Plotted {filename}")


def heatmap(logdir: str,
            mode: str = "single",
            skip_plot: bool = False,
            freq: int = 100,
            framerate: int = 6,
            gen: int = None,
            kiva: bool = False,
            vmax: float = None,
            heatmap_only: bool = False):
    """Plots the heatmaps for archives in a logdir.

    Args:
        logdir: Path to experiment logging directory.
        mode:
          - "single": plot the 2D archive and save to logdir /
            `heatmap_archive_{gen}.{pdf,png,svg}`
          - "single-3D": plot the 3D archive and save to logdir /
            `heatmap_archive_{gen}.{pdf,png,svg}`
          - "video": plot every `freq` generations and save to the directory
            logdir / `heatmap_archive`; logdir / `heatmap_archive.mp4` is also
            created from these images with ffmpeg.
        skip_plot: Skip plotting the heatmaps and just make the video. Only
            applies to "video" mode.
        freq: Frequency (in terms of generations) to plot heatmaps for video.
            Only applies to "video" mode.
        framerate: Framerate for the video. Only applies to "video" mode.
        gen: Generation to plot -- only applies to "single" mode.
            None indicates the final gen.
        mario: Pass this to activate several special settings for Mario.
        maze: Pass this to activate several special settings for Maze.
    """
    logdir = load_experiment(logdir)

    # if len(gin.query_parameter("GridArchive.dims")) != 2:
    #     logger.error("Heatmaps not supported for non-2D archives")
    #     return

    # Decide upper bound of heatmap based on config
    if vmax is None:
        sol_dim = gin.query_parameter("Manager.sol_size")
        agent_num = gin.query_parameter("WarehouseManager.agent_num")
        w_mode = gin.query_parameter("WarehouseManager.w_mode")
        # if sol_dim == 108:
        #     if 0 < agent_num <= 10:
        #         vmax = 1.2
        #     elif 10 < agent_num <= 20:
        #         vmax = 2
        #     elif 20 < agent_num <= 30:
        #         vmax = 2.8
        #     elif 30 < agent_num <= 40:
        #         vmax = 3.6
        #     elif 40 < agent_num <= 50:
        #         vmax = 4.3
        #     elif 50 < agent_num <= 60:
        #         vmax = 4.2
        #     else:
        #         vmax = 5

        # if sol_dim == 204:
        #     if 0 < agent_num <= 20:
        #         vmax = 1.8
        #     elif 20 < agent_num <= 40:
        #         vmax = 3.2
        #     elif 40 < agent_num <= 60:
        #         vmax = 2.8
        #     elif 60 < agent_num <= 80:
        #         vmax = 3.6
        #     elif 80 < agent_num <= 100:
        #         vmax = 6
        #     elif 100 < agent_num <= 120:
        #         vmax = 4.2
        #     else:
        #         vmax = 4

        # if sol_dim == 1056:
        #     vmax = 6.5

        # if not w_mode:
        #     vmax += 1.5

    vmin = 0
    vmax = 7

    # Plotting arguments for grid_archive_heatmap.
    plot_kwargs = {
        "square": False,
        "cmap": "viridis",
        "pcm_kwargs": {
            # Looks much better in PDF viewers because the heatmap is not drawn
            # as individual rectangles. See here:
            # https://stackoverflow.com/questions/27092991/white-lines-in-matplotlibs-pcolor
            "rasterized": True,
        },
        "vmin": vmin,
        "vmax": vmax,
        "plot_color_bar": True,
    }
    if heatmap_only:
        plot_kwargs["plot_color_bar"] = False

    total_gens = load_metrics(logdir).total_itrs
    gen = total_gens if gen is None else gen

    # surrogate_total_gens = gin.query_parameter("Manager.inner_itrs")

    if mode == "single":
        plot_generation(
            mode,
            logdir,
            gen,
            plot_kwargs,
            [
                f"heatmap_archive_{gen}.pdf",
                f"heatmap_archive_{gen}.png",
                f"heatmap_archive_{gen}.svg",
            ],
            kiva,
            heatmap_only,
        )
    elif mode == "single-3D":
        plot_generation_3D(
            mode,
            logdir,
            gen,
            plot_kwargs,
            [
                f"heatmap_archive_{gen}.pdf",
                f"heatmap_archive_{gen}.png",
                f"heatmap_archive_{gen}.svg",
            ],
            kiva,
        )
    elif mode == "video":  # pylint: disable = too-many-nested-blocks
        if not skip_plot:
            # Remove existing heatmaps.
            shutil.rmtree(logdir.pdir("heatmap_archive/"), ignore_errors=True)

            digits = int(np.ceil(np.log10(total_gens + 1)))
            for g in range(total_gens + 1):  # 0...total_gens
                try:
                    if g % freq == 0 or g == total_gens:
                        plot_generation(
                            mode,
                            logdir,
                            g,
                            plot_kwargs,
                            [f"heatmap_archive/{g:0{digits}}.png"],
                            kiva,
                        )
                except ValueError as e:
                    logger.error(
                        "ValueError caught. Have you tried setting the max "
                        "objective in objectives/__init__.py ?")
                    raise e

        # The extra options make the video web-compatible - see
        # https://gist.github.com/Vestride/278e13915894821e1d6f
        os.system(f"""\
ffmpeg -an -r {framerate} -i "{logdir.file('heatmap_archive/%*.png')}" \
    -vcodec libx264 \
    -pix_fmt yuv420p \
    -profile:v baseline \
    -level 3 \
    "{logdir.file('heatmap_archive.mp4')}" \
    -y \
""")
    else:
        raise ValueError(f"Unknown mode '{mode}'")


if __name__ == "__main__":
    fire.Fire(heatmap)

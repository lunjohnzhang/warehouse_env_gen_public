"""Utilities for other postprocessing scripts.

Note that most of these functions require that you first call `load_experiment`
so that gin configurations are loaded properly.
"""
import os
import pickle as pkl
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

import gin
import numpy as np
from logdir import LogDir

from ribs.visualize import _retrieve_cmap

# Including this makes gin config work because main imports (pretty much)
# everything.
import env_search.main  # pylint: disable = unused-import
from env_search.archives import GridArchive
from env_search.utils.deprecation import DEPRECATED_OBJECTS
from env_search.utils.metric_logger import MetricLogger

algo_name_map = {
    "Dummy": "DPP",
    "RHCR": "RHCR",
}


def load_experiment(logdir: str) -> LogDir:
    """Loads gin configuration and logdir for an experiment.

    Intended to be called at the beginning of an analysis script.

    Args:
        logdir: Path to the experiment's logging directory.
    Returns:
        LogDir object for the directory.
    """
    gin.clear_config()  # Erase all previous param settings.
    gin.parse_config_file(Path(logdir) / "config.gin",
                          skip_unknown=DEPRECATED_OBJECTS)
    logdir = LogDir(gin.query_parameter("experiment.name"), custom_dir=logdir)
    return logdir


def load_metrics(logdir) -> MetricLogger:
    return MetricLogger.from_json(logdir.file("metrics.json"))


def load_archive_from_history(logdir: LogDir, individual=False) -> GridArchive:
    """Generator that produces archives loaded from archive_history.pkl.

    Note that these archives will only contain objectives and BCs.

    Pass `individual` to indicate that the archive should be yielded after each
    solution is inserted into the archive, rather than only at the end of each
    iteration / generation.

    WARNING: Be careful that the history only recorded solutions that were
    inserted into the archive successfully, so many solutions are excluded.
    """
    archive_type = str(gin.query_parameter("Manager.archive_type"))
    if archive_type == "@GridArchive":
        # Same construction as in Manager.
        # pylint: disable = no-value-for-parameter
        archive = GridArchive(seed=42, dtype=np.float32)
    else:
        raise TypeError(f"Cannot handle archive type {archive_type}")
    archive.initialize(0)  # No solutions.

    with logdir.pfile("archive_history.pkl").open("rb") as file:
        archive_history = pkl.load(file)

    yield archive  # Start with empty archive.
    for gen_history in archive_history:
        archive.new_history_gen()
        for obj, bcs in gen_history:
            archive.add([], obj, bcs, None)  # No solutions, no metadata.
            if individual:
                yield archive
        if not individual:
            yield archive


def load_surrogate_archive(
        logdir: LogDir,
        outer_iter: int = -1,  # -1 for the last one
        # individual: bool = False,
) -> GridArchive:
    """Generator that produces archives loaded from archive_history.pkl.

    Note that these archives will only contain objectives and BCs.

    Pass `individual` to indicate that the archive should be yielded after each
    solution is inserted into the archive, rather than only at the end of each
    iteration / generation.

    WARNING: Be careful that the history only recorded solutions that were
    inserted into the archive successfully, so many solutions are excluded.
    """
    archive_type = str(gin.query_parameter("Manager.archive_type"))
    if archive_type == "@GridArchive":
        # Same construction as in Manager.
        # pylint: disable = no-value-for-parameter
        surrogate_archive = GridArchive(seed=42, dtype=np.float32)
    else:
        raise TypeError(f"Cannot handle archive type {archive_type}")
    surrogate_archive.initialize(0)  # No solutions.

    # Split archive and downsample archive, if any
    save_dir = logdir.dir("surrogate_archive")
    surrogate_archive_paths = []
    downsample_archive_paths = []
    for surrogate_archive_path in os.listdir(save_dir):
        archive_path_full = os.path.join(save_dir, surrogate_archive_path)
        if surrogate_archive_path.startswith("downsample"):
            downsample_archive_paths.append(archive_path_full)
        else:
            surrogate_archive_paths.append(archive_path_full)

    surrogate_archive_paths = sorted(surrogate_archive_paths)
    surrogate_archive_to_plot = surrogate_archive_paths[outer_iter]
    # breakpoint()
    with open(surrogate_archive_to_plot, "rb") as file:
        surrogate_archive_df = pkl.load(file)

    downsample_archive_to_plot = None
    downsample_archive = None
    if len(downsample_archive_paths) > 0:
        downsample_archive = GridArchive(seed=42, dtype=np.float32)
        downsample_archive.initialize(0)

        downsample_archive_paths = sorted(downsample_archive_paths)
        downsample_archive_to_plot = downsample_archive_paths[outer_iter]
        with open(downsample_archive_to_plot, "rb") as file:
            downsample_archive_df = pkl.load(file)

    surrogate_archive.new_history_gen()
    for _, row in surrogate_archive_df.iterrows():
        obj = row["objective"]
        bcs = [row["behavior_0"], row["behavior_1"]]
        surrogate_archive.add([], obj, bcs, None)

    if downsample_archive is not None:
        downsample_archive.new_history_gen()
        for _, row in downsample_archive_df.iterrows():
            obj = row["objective"]
            bcs = [row["behavior_0"], row["behavior_1"]]
            downsample_archive.add([], obj, bcs, None)

    return surrogate_archive, downsample_archive


def load_archive_gen(logdir: LogDir, gen: int) -> GridArchive:
    """Loads the archive at a given generation; works for ME-ES too."""
    itr = iter(load_archive_from_history(logdir))
    for _ in range(gen + 1):
        archive = next(itr)
    return archive


def get_color(map_from, algo_name, n_agents_opt=None):
    if map_from == "DSAGE" or map_from == "Optimized Layout":
        if algo_name == "RHCR":
            if n_agents_opt is None:
                return "orange"
            elif n_agents_opt == 150: # large w with 150 agents
                return "olive"
            elif n_agents_opt == 60: # small r with 60 agents
                return "gray"
            elif n_agents_opt == 40: # small r with 40 agents
                return "lime"
            else:
                return "orange"
        elif algo_name == "Dummy":
            return "green"
    elif map_from == "MAP-Elites":
        if algo_name == "RHCR":
            return "red"
        elif algo_name == "Dummy":
            return "cyan"
    elif map_from == "Human" or map_from == "Human-designed Layout":
        if algo_name == "RHCR":
            return "blue"
        elif algo_name == "Dummy":
            return "purple"



def grid_archive_heatmap(archive,
                         ax=None,
                         transpose_bcs=False,
                         cmap="magma",
                         square=False,
                         vmin=None,
                         vmax=None,
                         pcm_kwargs=None,
                         plot_color_bar=True,
                         cbar_ticklabel_fontsize=15):
    """Plots heatmap of a :class:`~ribs.archives.GridArchive` with 2D behavior
    space.

    Essentially, we create a grid of cells and shade each cell with a color
    corresponding to the objective value of that cell's elite. This method uses
    :func:`~matplotlib.pyplot.pcolormesh` to generate the grid. For further
    customization, pass extra kwargs to :func:`~matplotlib.pyplot.pcolormesh`
    through the ``pcm_kwargs`` parameter. For instance, to create black
    boundaries of width 0.1, pass in ``pcm_kwargs={"edgecolor": "black",
    "linewidth": 0.1}``.

    Examples:
        .. plot::
            :context: close-figs

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import GridArchive
            >>> from ribs.visualize import grid_archive_heatmap
            >>> # Populate the archive with the negative sphere function.
            >>> archive = GridArchive([20, 20], [(-1, 1), (-1, 1)])
            >>> archive.initialize(solution_dim=2)
            >>> for x in np.linspace(-1, 1, 100):
            ...     for y in np.linspace(-1, 1, 100):
            ...         archive.add(solution=np.array([x,y]),
            ...                     objective_value=-(x**2 + y**2),
            ...                     behavior_values=np.array([x,y]))
            >>> # Plot a heatmap of the archive.
            >>> plt.figure(figsize=(8, 6))
            >>> grid_archive_heatmap(archive)
            >>> plt.title("Negative sphere function")
            >>> plt.xlabel("x coords")
            >>> plt.ylabel("y coords")
            >>> plt.show()


    Args:
        archive (GridArchive): A 2D GridArchive.
        ax (matplotlib.axes.Axes): Axes on which to plot the heatmap. If None,
            the current axis will be used.
        transpose_bcs (bool): By default, the first BC in the archive will
            appear along the x-axis, and the second will be along the y-axis. To
            switch this (i.e. to transpose the axes), set this to True.
        cmap (str, list, matplotlib.colors.Colormap): Colormap to use when
            plotting intensity. Either the name of a colormap, a list of RGB or
            RGBA colors (i.e. an Nx3 or Nx4 array), or a colormap object.
        square (bool): If True, set the axes aspect ratio to be "equal".
        vmin (float): Minimum objective value to use in the plot. If None, the
            minimum objective value in the archive is used.
        vmax (float): Maximum objective value to use in the plot. If None, the
            maximum objective value in the archive is used.
        pcm_kwargs (dict): Additional kwargs to pass to
            :func:`~matplotlib.pyplot.pcolormesh`.
    Raises:
        ValueError: The archive is not 2D.
    """
    if archive.behavior_dim != 2:
        raise ValueError("Cannot plot heatmap for non-2D archive.")

    # Try getting the colormap early in case it fails.
    cmap = _retrieve_cmap(cmap)

    # Retrieve data from archive.
    lower_bounds = archive.lower_bounds
    upper_bounds = archive.upper_bounds
    x_dim, y_dim = archive.dims
    x_bounds = archive.boundaries[0]
    y_bounds = archive.boundaries[1]

    # Color for each cell in the heatmap.
    colors = np.full((y_dim, x_dim), np.nan)
    for elite in archive:
        colors[elite.idx[1], elite.idx[0]] = elite.obj

    if transpose_bcs:
        # Since the archive is 2D, transpose by swapping the x and y boundaries
        # and by flipping the bounds (the bounds are arrays of length 2).
        x_bounds, y_bounds = y_bounds, x_bounds
        lower_bounds = np.flip(lower_bounds)
        upper_bounds = np.flip(upper_bounds)
        colors = colors.T

    # Initialize the axis.
    ax = plt.gca() if ax is None else ax
    ax.set_xlim(lower_bounds[0], upper_bounds[0])
    ax.set_ylim(lower_bounds[1], upper_bounds[1])

    if square:
        ax.set_aspect("equal")

    # Create the plot.
    pcm_kwargs = {} if pcm_kwargs is None else pcm_kwargs
    objectives = archive.as_pandas().batch_objectives()
    vmin = np.min(objectives) if vmin is None else vmin
    vmax = np.max(objectives) if vmax is None else vmax
    t = ax.pcolormesh(x_bounds,
                      y_bounds,
                      colors,
                      cmap=cmap,
                      vmin=vmin,
                      vmax=vmax,
                      **pcm_kwargs)

    # Create the colorbar.
    if plot_color_bar:
        cbar = ax.figure.colorbar(t, ax=ax, pad=0.1)
        cbar.set_ticks([vmin, np.mean([vmin, vmax]), vmax])
        cbar.set_ticklabels([vmin, np.mean([vmin, vmax]), vmax])
        cbar.ax.tick_params(labelsize=cbar_ticklabel_fontsize)
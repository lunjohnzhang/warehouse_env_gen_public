import os
import fire
from typing import List
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import colors
from env_search import MAP_DIR
from env_search.utils import (kiva_obj_types, kiva_env_str2number,
                              kiva_env_number2str, read_in_kiva_map,
                              KIVA_ROBOT_BLOCK_WIDTH)
from env_search.utils import set_spines_visible


def visualize_kiva(
    env_np: np.ndarray,
    filenames: List = None,
    store_dir: str = MAP_DIR,
    dpi: int = 300,
    ax: plt.Axes = None,
):
    """
    Visualize kiva layout. Will store image under env_search/map/

    Args:
        env_np: layout in numpy format
    """
    n_row, n_col = env_np.shape
    save = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(n_col, n_row))
        save = True
    cmap = colors.ListedColormap(
        ['white', 'black', 'deepskyblue', 'orange', 'fuchsia'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # heatmap = plt.pcolor(np.array(data), cmap=cmap, norm=norm)
    # plt.colorbar(heatmap, ticks=[0, 1, 2, 3])
    sns.heatmap(
        env_np,
        square=True,
        cmap=cmap,
        norm=norm,
        ax=ax,
        cbar=False,
        rasterized=True,
        annot_kws={"size": 30},
        linewidths=1,
        linecolor="black",
        xticklabels=False,
        yticklabels=False,
    )

    set_spines_visible(ax)
    ax.figure.tight_layout()

    if save:
        ax.margins(x=0, y=0)
        for filename in filenames:
            fig.savefig(
                os.path.join(store_dir, filename),
                dpi=dpi,
                bbox_inches='tight',
                # pad_inches=0,
            )
        plt.close('all')


def main(map_filepath, store_dir=MAP_DIR):
    kiva_map, map_name = read_in_kiva_map(map_filepath)
    visualize_kiva(kiva_env_str2number(kiva_map),
                   store_dir=store_dir,
                   filenames=[f"{map_name}.png"])


if __name__ == '__main__':
    fire.Fire(main)
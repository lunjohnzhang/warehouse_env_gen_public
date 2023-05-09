"""Miscellaneous project-wide utilities."""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from env_search import MAP_DIR


# 4 object types for kiva map:
# '.' (0) : empty space
# '@' (1): obstacle (shelf)
# 'e' (2): endpoint (point around shelf)
# 'r' (3): robot start location (not searched)
# 's' (4): one of 'r'
# 'w' (5): workstation
# Note 1: only the first 2 or 3 objects are searched by QD
# Note 2: s (r_s) is essentially one of r s.t. in milp can make the graph
# connected
kiva_obj_types = ".@ersw"
KIVA_ROBOT_BLOCK_WIDTH = 4
KIVA_WORKSTATION_BLOCK_WIDTH = 2
MIN_SCORE = 0

KIVA_ROBOT_BLOCK_HEIGHT = 4

def format_env_str(env_str):
    """Format the env from List[str] to pure string separated by \n """
    return "\n".join(env_str)

def kiva_env_str2number(env_str):
    """
    Convert kiva env in string format to np int array format.

    Args:
        env_str (List[str]): kiva env in string format

    Returns:
        env_np (np.ndarray)
    """
    env_np = []
    for row_str in env_str:
        # print(row_str)
        row_np = [kiva_obj_types.index(tile) for tile in row_str]
        env_np.append(row_np)
    return np.array(env_np, dtype=int)


def kiva_env_number2str(env_np):
    """
    Convert kiva env in np int array format to str format.

    Args:
        env_np (np.ndarray): kiva env in np array format

    Returns:
        env_str (List[str])
    """
    env_str = []
    n_row, n_col = env_np.shape
    for i in range(n_row):
        curr_row = []
        for j in range(n_col):
            curr_row.append(kiva_obj_types[env_np[i, j]])
        env_str.append("".join(curr_row))
    return env_str

def flip_one_r_to_s(env_np):
    """
    Change one of 'r' in the env to 's' for milp
    """
    all_r = np.argwhere(env_np == kiva_obj_types.index("r"))
    if len(all_r) == 0:
        raise ValueError("No 'r' found")
    to_replace = all_r[0]
    env_np[tuple(to_replace)] = kiva_obj_types.index('s')
    return env_np

def flip_tiles(env_np, from_tile, to_tile):
    """Replace ALL occurance of `from_tile` to `flip_target` in `to_tile"""
    all_from_tiles = np.where(env_np == kiva_obj_types.index(from_tile))
    if len(all_from_tiles[0]) == 0:
        raise ValueError(f"No '{from_tile}' found")
    env_np[all_from_tiles] = kiva_obj_types.index(to_tile)
    return env_np

def read_in_kiva_map(map_filepath):
    """
    Read in kiva map and return in str format
    """
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)
        raw_env = raw_env_json["layout"]
        name = raw_env_json["name"]
    return raw_env, name


def set_spines_visible(ax: plt.Axes):
    for pos in ["top", "right", "bottom", "left"]:
        ax.spines[pos].set_visible(True)
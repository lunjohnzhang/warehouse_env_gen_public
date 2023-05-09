import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
CONFIG_DIR = os.path.join(_parent_dir, "config")
LOG_DIR = os.path.join(_parent_dir, "logs")
MAP_DIR = os.path.join(_parent_dir, "maps")
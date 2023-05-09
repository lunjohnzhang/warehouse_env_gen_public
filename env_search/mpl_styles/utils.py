"""Utils for handling styles."""
import importlib.resources

import env_search.mpl_styles


def mpl_style_file(name: str):
    """Returns a context manager for using the given style file.

    See here for more info:
    https://stackoverflow.com/questions/6028000/how-to-read-a-static-file-from-inside-a-python-package

    Example:

        from src.mpl_styles.utils import mpl_style_file

        with mpl_style_file("simple.mplstyle") as f:
            with plt.style.context(f):
                plt.plot(...)
    """
    return importlib.resources.path(env_search.mpl_styles, name)

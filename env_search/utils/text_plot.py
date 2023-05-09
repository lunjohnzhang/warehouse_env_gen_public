"""Functions for creating a text-based scatterplot.

Adapted from terminalplot - see
https://github.com/kressi/terminalplot/blob/master/terminalplot/terminalplot.py
"""
from typing import Sequence


def _scale(x, length):
    """Scale points in 'x', such that distance between max(x) and min(x) equals
    to 'length'.

    min(x) will be moved to 0.
    """
    max_x, min_x = max(x), min(x)
    s = (float(length - 1) /
         (max_x - min_x) if x and max_x - min_x != 0 else length)
    return [int((i - min_x) * s) for i in x]


def text_plot(x: Sequence,
              y: Sequence,
              width: int = 80,
              height: int = 20) -> str:
    """Creates a str scatterplot of the given x and y.

    Args:
        x: List of x coords
        y: List of y coords
        width: Number of columns in the plot (each column is one char wide).
        height: Number of rows in the plot (each row is one char tall).
    Returns:
        Multi-line str that can be printed to the console.
    """
    x, y = list(x), list(y)

    height -= 3  # Caption offset (1 row), and borders (2 rows)
    width -= 2  # Border offset (2 columns)

    # Scale points such that they fit on canvas.
    x_scaled = _scale(x, width)
    y_scaled = _scale(y, height)

    # Create empty canvas.
    top_border = ['+'] + ['-'] * width + ['+']
    lines = [['|'] + [' ' for _ in range(width)] + ['|'] for _ in range(height)]
    canvas = [top_border] + lines + [top_border]

    # Add scaled points to canvas
    for ix, iy in zip(x_scaled, y_scaled):
        # -1 for the offset, +1 because of border.
        canvas[height - iy - 1 + 1][ix + 1] = '*'

    canvas = ["".join(row) for row in canvas]
    canvas.append(f"x: [{min(x)}, {max(x)}]  y: [{min(y)}, {max(y)}]")
    return "\n".join(canvas)

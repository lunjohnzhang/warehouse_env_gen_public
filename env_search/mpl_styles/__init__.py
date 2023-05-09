"""Styles for Matplotlib."""
from matplotlib.colors import ListedColormap

# Qualitative colormap that (should) be color-blind friendly. See
# https://personal.sron.nl/~pault/ for more accessible color schemes.
QUALITATIVE_COLORS = (
    '#77AADD',
    '#EE8866',
    '#44BB99',
    '#FFAABB',
    '#99DDFF',
    '#BBCC33',
    '#EEDD88',
    '#AAAA00',
)
QualitativeMap = ListedColormap(QUALITATIVE_COLORS)

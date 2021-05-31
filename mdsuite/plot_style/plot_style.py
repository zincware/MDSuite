"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.
"""

import matplotlib.pyplot as plt
from matplotlib import checkdep_usetex


def style():
    """
    Define the plot style.
    Returns
    -------
    Sets a style.
    """
    font_size = 36
    params = {
        'font.family': 'STIXGeneral',
        'font.serif': 'Computer Modern',
        'font.sans-serif': 'Computer Modern Sans serif',
        'font.size': font_size,
        'axes.labelsize': font_size,
        'legend.fontsize': font_size - 5,
        'xtick.labelsize': font_size - 5,
        'ytick.labelsize': font_size - 5,
        'axes.grid': False,
        'xtick.top': False,
        'ytick.right': False,
        'figure.figsize': [16, 9],
        'text.usetex': True,
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "pgf.preamble": "\\usepackage[utf8x]{inputenc}\n \\usepackage[T1]{fontenc}",
        'axes.linewidth': 1.5,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "lines.linewidth": 3.0,
        "lines.markersize": 12.0,
        "lines.markeredgewidth": 2.0,
        "legend.frameon": False,
        "legend.edgecolor": (1, 1, 1),
        "legend.framealpha": 0.5,
        "legend.fancybox": "False",
        'legend.numpoints': 1,
        "xtick.major.size": 10.0,
        "xtick.major.width": 2.5,
        "xtick.minor.size": 6.0,
        "xtick.minor.width": 1.5,
        "ytick.major.size": 10.0,
        "ytick.major.width": 2.5,
        "ytick.minor.size": 6.0,
        "ytick.minor.width": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        'axes.autolimit_mode': 'round_numbers',
        'figure.autolayout': True,
        'axes.xmargin': 0.03,
        'axes.ymargin': 0.03,
        'axes.spines.top': False,
        'axes.spines.right': False
        # 'figure.constrained_layout.use': True
    }

    usetex = checkdep_usetex(True)  # check if latex is available
    if not usetex:  # if not, just set it to False, and do not use latex to print figures.
        params['text.usetex'] = False

    return params


def apply_style():
    """
    Apply the style to the kernel.
    Returns
    -------
    Sets plotting styles for the kernel.
    """
    params = style()
    plt.style.use(params)


def get_markers():
    """
    Prepare a list of markers for use in the styling.
    Returns
    -------
    Returns a list of markers.
    """
    # list of markers
    return [".", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D",
            "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

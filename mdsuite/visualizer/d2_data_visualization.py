"""
MDSuite: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
"""
import numpy as np
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.io import output_notebook, output_file
from bokeh.models import HoverTool
from typing import Union, List
from mdsuite.utils import config


class DataVisualizer2D:
    """
    Visualizer for two-dimensional data.
    """

    def __init__(self, title: str):
        """
        Constructor for the data visualizer.

        Parameters
        ----------
        title : str
                title of the plot.
        """
        if config.jupyter:
            output_notebook()
        else:
            output_file(f"{title}.html", title=title)

    def construct_plot(
        self,
        x_data: Union[list, np.ndarray],
        y_data: Union[list, np.ndarray],
        x_label: str,
        y_label: str,
        title: str,
        layouts: List = None,
    ) -> figure:
        """
        Generate a plot.

        Parameters
        ----------
        x_data : Union[list, np.ndarray, tf.Tensor]
                data to plot along the x axis.
        y_data : Union[list, np.ndarray, tf.Tensor]
                data to plot along the y axis.
        x_label : str
                label for the x axis
        y_label : str
                label of the y axis.
        title : str
                name of the specific plot.
        Returns
        -------
        figure : figure
                A bokeh figure object.
        """
        fig = figure(x_axis_label=x_label, y_axis_label=y_label, sizing_mode='stretch_both')
        fig.line(x_data, y_data, legend_label=title)
        fig.add_tools(HoverTool())
        if layouts is not None:
            for item in layouts:
                fig.add_layout(item)

        return fig

    def grid_show(self, figures: list):
        """
        Display a list of figures in a grid.

        Parameters
        ----------
        figures : list
                A list of figures to display.

        Returns
        -------

        """
        grid = gridplot(figures, ncols=3, sizing_mode="scale_both")
        show(grid)

"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Visualize a simulation.
"""
import numpy as np
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.io import output_notebook, output_file
from bokeh.models import HoverTool

from typing import Union


class DataVisualizer2D:
    """
    Visualizer for two-dimensional data.

    Attributes
    ----------
    data : np.ndarray
            Zipped numpy array.
    x_label : str
                x label as a string.
    y_label : str
            y label as a string.
    """

    def __init__(self, title: str):
        """
        Constructor for the data visualizer.

        Parameters
        ----------
        title : str
                title of the plot.
        """
        output_notebook()
        output_file(f"{title}.html", title=title)
        pass

    def construct_plot(
            self,
            x_data: Union[list, np.ndarray],
            y_data: Union[list, np.ndarray],
            x_label: str,
            y_label: str,
            title: str
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
        fig = figure(
            title=title, x_axis_label=x_label, y_axis_label=y_label
        )
        fig.line(x_data, y_data)
        fig.add_tools(HoverTool())

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
        grid = gridplot(figures, ncols=3)
        show(grid)

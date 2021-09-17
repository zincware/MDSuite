"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Visualize a simulation.
"""
import numpy as np
import plotly.graph_objects as go
from bokeh.plotting import figure, show


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

    def __init__(self,
                 x_data: np.ndarray,
                 y_data: np.ndarray,
                 x_label: str,
                 y_label: str,
                 title: str):
        """
        Constructor for the data visualizer.

        Parameters
        ----------
        x_data : np.ndarray
                x axis data to plot.
        y_data : np.ndarray
                y axis data to plot.
        x_label : str
                x label as a string.
        y_label : str
                y label as a string.
        title : str
                title of the plot.
        """
        self.data = np.array(list(zip(x_data, y_data)))
        self.x_label = x_label
        self.y_label = y_label
        self.title = title

    def plot(self):
        """

        Returns
        -------

        """
        fig = figure(
            title=self.title, x_axis_label=self.x_label, y_axis_label=self.y_label
        )


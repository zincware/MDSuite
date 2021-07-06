"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: A base class for time series analysis
"""
from __future__ import annotations
from mdsuite.database.simulation_database import Database
import matplotlib.pyplot as plt

import tensorflow as tf

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mdsuite import Experiment


class TimeSeries:
    def __init__(self, experiment: Experiment):
        """

        Parameters
        ----------
        experiment: Experiment
            The parent experiment class to perform the time series operation on
        """
        self.experiment = experiment

        self.loaded_property = None
        self.fig_labels = {
            "x": None,
            "y": None
        }

        # Properties
        self._database = None
        self._data = None

    def __call__(self):
        self.plot()

    @property
    def database(self):
        """Get the database"""
        if self._database is None:
            self._database = Database(name=self.experiment.database_path + r"\database.hdf5")
        return self._database

    @property
    def data(self):
        """Get the data for all species and timesteps for the loaded_property"""
        if self._data is None:
            self._data = tf.concat(
                [self.database.load_data([f'{species}/{self.loaded_property}']) for species in self.experiment.species],
                axis=0
            )
        return self._data

    def plot(self):
        """Plot the data over timesteps"""
        fig, ax = plt.subplots()
        ax.plot(tf.einsum("atx -> t", self.data))
        # performe a reduce sum over atoms "a" and property dimension "x" to yield time steps "t"
        ax.set_xlabel(self.fig_labels['x'])
        ax.set_ylabel(self.fig_labels['y'])
        fig.show()

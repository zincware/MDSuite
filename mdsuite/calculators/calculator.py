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
Parent class for the calculators.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

if TYPE_CHECKING:
    from mdsuite import Experiment

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


def call(*args, **kwargs):
    pass


@dataclass
class ComputationResults:
    """
    A wrapper class for the results of a computation.

    This class is returned when data is loaded from the SQL database.
    """

    data: dict = field(default_factory=dict)
    subjects: dict = field(default_factory=list)


class Calculator:
    """
    Parent class for analysis modules

    Attributes
    ----------
    experiment : Experiment
                Experiment for which the calculator will be run.
    experiments : List[Experiment]
            List of experiments on which to run the calculator.
    plot : bool
            If true, the results will be plotted.
    system_property: bool (default = False)
            If the calculator returns a value for the whole system such as ionic
            conductivity or viscosity as opposed to a species-specific number.
    experimental : bool (default = False)
            If true, a warning is raised upon calling this calculator with more
            information about why it is experimental.
    selected_species: tuple
            Species currently being studied in a specific loop.
    analysis_name: str
            Name of the analysis to store in the database.
    time : np.ndarray
            Time array over which to integrate and plot.
    plotter : DataVisualizer2D
            Data visualizer class for use in the plotting.
    result_keys : list
            keys to use when storing the results. e.g.
            ["diffusion_coefficient", "uncertainty"]
    result_series_keys : list
            keys to use when storing series results e.g.
            ["time", "msd"]
    prefactor : float (optional)
            can be set if the same pre-factor is required many times.
    x_label : str
            x-label for the plots.
    y_label : str
            y-label for the plots.
    _dtype : object = tf.float64
            dtype required by the analysis.
    plot_array : list
            A list of plot objects to be show together at the end of the
            species loop.
    """

    def __init__(
        self, experiment: Experiment = None, experiments: List[Experiment] = None
    ):
        """
        Constructor for the calculator class.

        Parameters
        ----------
        experiment : Experiment
                Experiment for which the calculator will be run.
        experiments : List[Experiment]
                List of experiments on which to run the calculator.
        """
        # Set upon instantiation of parent class
        # super().__init__(experiment)
        # NOTE: if the calculator accepts more than just experiment/experiments
        #  in the init the @call decorator has to be changed!
        self.experiment: Experiment = experiment
        self.experiments: List[Experiment] = experiments
        self._queued_data = []
        # Setting the experiment value supersedes setting experiments
        if self.experiment is not None:
            self.experiments = [self.experiment]

        self.plot = False

        # SQL data attributes.
        self.result_keys = None
        self.result_series_keys = None
        self.analysis_name = None
        self.selected_species = None
        self.stored_parameters = None

        # Calculator attributes
        self.system_property = False
        self.experimental = False
        self.time = None
        self.prefactor = None

        # Data attributes
        self._dtype: object = tf.float64

        # Plotting attributes
        self.plotter = None
        self.x_label = None
        self.y_label = None
        self.plot_array = []

    def adopt_experiment_attributes(self, simulation_attributes):
        """
        Collect some important attributes from the experiment for
        internal use.

        Parameters
        ----------
        simulation_attributes : object
                Collection of simulation attributes that are required in the
                calculators.

        Returns
        -------

        """
        self.experiment_timestep = simulation_attributes.time_step
        self.experiment_sample_rate = simulation_attributes.sample_rate

    def __call__(
        self,
    ):
        """
        Call the calculator.
        """
        self.run_calculator()  # perform the computation.

        return self._queued_data

    def prepare_calculation(self):
        """
        Helper method for parameters that need to be computed after the experiment
        attributes are exposed to the calculator.
        Returns
        -------

        """
        pass

    @property
    def dtype(self):
        """Get the dtype used for the calculator"""
        return self._dtype

    def run_visualization(
        self, x_data: np.ndarray, y_data: np.ndarray, title: str, layouts: object = None
    ):
        """
        Run a visualization session on the data.

        Parameters
        ----------
        layouts : object
                Additional plot features that may be added.
                See https://docs.bokeh.org/en/latest/docs/reference/models.html for
                more information.
        x_data : np.ndarray
                Data to be plotted along the x axis
        y_data : np.ndarray
                Data to be plotted along the y-axis
        title : str
                Title of the analysis.

        Returns
        -------
        Updates the plot array with a Bokeh plot object.
        """
        self.plot_array.append(
            self.plotter.construct_plot(
                x_data=x_data,
                y_data=y_data,
                title=title,
                x_label=self.x_label,
                y_label=self.y_label,
                layouts=layouts,
            )
        )

    def run_calculator(self):
        """
        Run the calculation. This should be implemented in each calculator.

        Returns
        -------

        """
        raise NotImplementedError

    def plot_data(self, data):
        """
        Plot the data coming from the database

        Parameters
        ----------
        data: db.Compution.data_dict
                associated with the current project
        """
        for selected_species, val in data.items():
            self.run_visualization(
                x_data=np.array(val[self.result_series_keys[0]]),
                y_data=np.array(val[self.result_series_keys[1]]),
                title=(
                    f"{selected_species}: {val[self.result_keys[0]]: 0.3E} +-"
                    f" {val[self.result_keys[1]]: 0.3E}"
                ),
            )

    def run_analysis(self):
        """
        Run the appropriate analysis
        """
        if self.experimental:
            log.warning(
                "This is an experimental calculator. Please see the "
                "documentation before using the results."
            )
        self.run_calculator()

    def queue_data(self, data, subjects):
        """Queue data to be stored in the database

        Parameters:
            data: dict
                A  dictionary containing all the data that was computed by the
                computation
            subjects: list
                A list of strings / subject names that are associated with the data,
                e.g. the pairs of the RDF
        """
        self._queued_data.append(ComputationResults(data=data, subjects=subjects))

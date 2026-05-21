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
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import mdsuite.database.scheme as db
from mdsuite.database.calculator_database import CalculatorDatabase
from mdsuite.visualizer.d2_data_visualization import DataVisualizer2D

if TYPE_CHECKING:
    from mdsuite import Experiment

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


class Calculator(CalculatorDatabase):
    """
    Parent class for analysis modules.

    Calculators are now standalone configuration objects. They are constructed
    without an experiment, then applied to one or more experiments via
    ``Calculator.run(experiment)`` (or, equivalently, ``experiment.run(calc)`` /
    ``project.run(calc)``).

    Attributes
    ----------
    experiment : Experiment
            The experiment currently being processed. Set transiently by
            :meth:`run`; ``None`` outside of a run. Concrete calculators may
            read this during :meth:`run_calculator` for trajectory and
            metadata access.
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

    def __init__(self):
        """Constructor for the calculator class.

        Subclasses should accept their user-facing arguments here (data_range,
        plot, correlation_time, ...) and store them. The calculator must not
        depend on any experiment at construction time.
        """
        super().__init__()
        self.experiment: Experiment = None

        self.plot = False

        # SQL data attributes.
        self.result_keys = None
        self.result_series_keys = None
        self.analysis_name = None
        self.selected_species = None

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

    @property
    def dtype(self):
        """Get the dtype used for the calculator."""
        return self._dtype

    def run(self, experiment: Experiment) -> db.Computation:
        """Apply this calculator to a single experiment.

        Replaces the previous ``@call`` decorator. Handles the full lifecycle:
        per-run state setup, database cache lookup, analysis, persistence, and
        optional plotting.

        Parameters
        ----------
        experiment : Experiment
            The experiment to analyze.

        Returns
        -------
        db.Computation
            The computation entry from the database (cached or freshly
            computed).
        """
        self.experiment = experiment
        try:
            self._reset_run_state()
            self._setup()

            data = self.get_computation_data()
            if data is None:
                self.prepare_db_entry()
                self.save_computation_args()
                self.run_analysis()
                self.save_db_data()
                # ``_handle_tau_values`` no longer mutates ``self.args`` —
                # it writes ``self.resolved_*`` instead — so the cache
                # lookup keys (which come from ``self.args``) stay stable
                # across the run and no rewind is needed.
                data = self.get_computation_data()

            if self.plot:
                self.plotter = DataVisualizer2D(
                    title=self.analysis_name, path=experiment.figures_path
                )
                self.plot_data(data.data_dict)
                self.plotter.grid_show(self.plot_array)

            return data
        finally:
            self.experiment = None

    def _reset_run_state(self):
        """Clear any transient state from a previous run.

        Calculators are reusable across experiments; this guarantees no
        bleed-through between sequential ``run()`` calls.
        """
        self._queued_data = []
        self.db_computation_attributes = []
        self.db_computation = None
        self.plot_array = []

    def _setup(self):
        """Hook for subclasses to compute derived state after ``self.experiment``
        is set.

        Default is a no-op. Override when the calculator needs to compute
        experiment-dependent state (defaults filled from species lists, derived
        time arrays, prefactors that depend on units, etc.).
        """
        pass

    def run_visualization(
        self, x_data: np.ndarray, y_data: np.ndarray, title: str, layouts: object = None
    ):
        """Run a visualization session on the data.

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
        """Run the calculation. Must be implemented in each concrete calculator.

        Concrete subclasses access experiment state via ``self.experiment``,
        which is set for the duration of :meth:`run`.
        """
        raise NotImplementedError

    def plot_data(self, data):
        """Plot the data coming from the database.

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
        """Run the appropriate analysis."""
        if self.experimental:
            log.warning(
                "This is an experimental calculator. Please see the "
                "documentation before using the results."
            )
        self.run_calculator()

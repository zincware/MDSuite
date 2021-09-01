"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Class for the calculation of the coordinated numbers
"""
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Union
from mdsuite.database.property_database import PropertiesDatabase
from mdsuite.database.database_scheme import SystemProperty
from mdsuite.utils.exceptions import NotApplicableToAnalysis
from mdsuite.calculators.calculator import Calculator
from typing import Iterable

log = logging.getLogger(__file__)


class KirkwoodBuffIntegral(Calculator):
    """
    Class for the calculation of the Kikrwood-Buff integrals

    Attributes
    ----------
    experiment : class object
                        Class object of the experiment.
    data_range : int (default=500)
                        Range over which the property should be evaluated. This is not applicable to the current
                        analysis as the full rdf will be calculated.
    x_label : str
                        How to label the x axis of the saved plot.
    y_label : str
                        How to label the y axis of the saved plot.
    analysis_name : str
                        Name of the analysis. used in saving of the tensor_values and figure.
    file_to_study : str
                        The tensor_values file corresponding to the rdf being studied.
    data_files : list
                        list of files to be analyzed.
    rdf = None : list
                        rdf tensor_values being studied.
    radii = None : list
                        radii tensor_values corresponding to the rdf.
    species_tuple : list
                        A list of species combinations being studied.

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.KirkwoodBuffIntegral()
    """

    def __init__(self, **kwargs):
        """
        Python constructor for the class

        Parameters
        ----------
        experiment : class object
                        Class object of the experiment.
        """

        super().__init__(**kwargs)
        self.file_to_study = None
        self.data_files = []
        self.rdf = None
        self.radii = None
        self.species_tuple = None
        self.kb_integral = None
        self.database_group = "Kirkwood_Buff_Integral"
        self.x_label = r"r ($\AA$)"
        self.y_label = r"$G(\mathbf{r})$"
        self.analysis_name = "Kirkwood-Buff_Integral"

        self.post_generation = True

    def __call__(self, plot=True, save=True, data_range=1, export: bool = False):
        """
        Doc string for this one.
        Parameters
        ----------
        plot : bool
                If true, the output will be displayed in a figure. This figure will also be saved.
        save : bool
                If true, the data from the analysis will be saved in the sql database
        data_range : int
                Default to 1 for this analysis
        export : bool
                If tue, csv files will be dumped after the analysis.

        Returns
        -------
        data:
            A dictionary of shape {experiment_name: data} for multiple len(experiments) > 1 or otherwise just data

        """
        out = {}
        for experiment in self.experiments:
            self.experiment = experiment
            self.update_user_args(
                plot=plot, save=save, data_range=data_range, export=export
            )

            if self.load_data:
                out.update(
                    {
                        self.experiment.experiment_name: self.experiment.export_property_data(
                            {"Analysis": self.analysis_name}
                        )
                    }
                )
            else:
                out.update({self.experiment.experiment_name: self.run_analysis()})

        if len(out) > 1:
            return out
        else:
            return out[self.experiment.experiment_name]

    def _autocorrelation_time(self):
        """
        Not needed in this analysis
        """
        raise NotApplicableToAnalysis

    def _calculate_kb_integral(self):
        """
        calculate the Kirkwood-Buff integral
        """

        self.kb_integral = []  # empty the integration tensor_values

        for i in range(1, len(self.radii)):
            self.kb_integral.append(
                4
                * np.pi
                * (
                    np.trapz(
                        (self.rdf[1:i] - 1) * (self.radii[1:i]) ** 2, x=self.radii[1:i]
                    )
                )
            )

    def run_post_generation_analysis(self):
        """
        Calculate the potential of mean-force and perform error analysis
        """

        for data in self._get_rdf_data():  # Loop over all existing RDFs
            self.species_tuple = "_".join(data.subjects)
            self.data_range = data.data_range

            self._load_rdf_from_file(data)  # load the tensor_values from it

            self._calculate_kb_integral()  # Integrate the rdf and calculate the KB integral

            # Plot if required
            if self.plot:
                plt.plot(
                    self.radii[1:], self.kb_integral, label=f"{self.species_tuple}"
                )
                self._plot_data(title=f"{self.analysis_name}_{self.species_tuple}")

            if self.save or self.export:
                data = [
                    {"x": x, "y": y} for x, y in zip(self.radii[1:], self.kb_integral)
                ]
                log.debug(f"Writing {self.analysis_name} to database!")
                self._update_properties_file(
                    {
                        "Property": self.system_property,
                        "Analysis": self.analysis_name,
                        "subjects": self.species_tuple.split("_"),
                        "data_range": self.data_range,
                        "data": data,
                    }
                )

    def _calculate_prefactor(self, species: Union[str, tuple] = None):
        """
        calculate the calculator pre-factor.

        Parameters
        ----------
        species : str
                Species property if required.
        Returns
        -------

        """
        raise NotImplementedError

    def _apply_operation(self, data, index):
        """
        Perform operation on an ensemble.

        Parameters
        ----------
        One tensor_values range of tensor_values to operate on.

        Returns
        -------

        """
        raise NotImplementedError

    def _apply_averaging_factor(self):
        """
        Apply an averaging factor to the tensor_values.
        Returns
        -------
        averaged copy of the tensor_values
        """
        raise NotImplementedError

    def _post_operation_processes(self, species: Union[str, tuple] = None):
        """
        call the post-op processes
        Returns
        -------

        """
        raise NotImplementedError

    def _update_output_signatures(self):
        """
        After having run _prepare managers, update the output signatures.

        Returns
        -------

        """
        raise NotImplementedError

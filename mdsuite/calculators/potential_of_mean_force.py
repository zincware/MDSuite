"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.
"""

""" Class for the calculation of the coordinated numbers

Summary
-------
The potential of mean-force is a measure of the binding strength between atomic species in a experiment. Mathematically
    one may write

    .. math::

        g(r) = e^{-\frac{w^{(2)}(r)}{k_{B}T}}

    Which, due to us having direct access to the radial distribution functions, compute as

    .. math::

        w^{(2)}(r) = -k_{B}Tln(g(r))
"""
import logging

import numpy as np
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from typing import Union

from mdsuite.database.properties_database import PropertiesDatabase
from mdsuite.database.database_scheme import SystemProperty

from mdsuite.utils.exceptions import *
from mdsuite.calculators.calculator import Calculator
from mdsuite.utils.meta_functions import golden_section_search
from mdsuite.utils.meta_functions import apply_savgol_filter
from mdsuite.utils.units import boltzmann_constant

log = logging.getLogger(__file__)


class PotentialOfMeanForce(Calculator):
    """
    Class for the calculation of the potential of mean-force

    The potential of mean-force is a measure of the binding strength between atomic species in a experiment.

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
    experiment.run_computation.PotentialOfMeanForce(savgol_order = 2, savgol_window_length = 17)
    """

    def __init__(self, experiment):
        """
        Python constructor for the class

        Parameters
        ----------
        experiment : class object
                        Class object of the experiment.
        """

        super().__init__(experiment)
        self.file_to_study = None
        self.rdf = None
        self.radii = None
        self.species_tuple = None
        self.pomf = None
        self.indices = None
        self.database_group = 'Potential_Of_Mean_Force'
        self.x_label = r'r ($\AA$)'
        self.y_label = r'$w^{(2)}(r)$'
        self.analysis_name = 'Potential_of_Mean_Force'
        self.post_generation = True

    def __call__(self, plot=True, save=True, data_range=1, export: bool = False,
                 savgol_order: int = 2, savgol_window_length: int = 17):
        """
        Python constructor for the class

        Parameters
        ----------
        plot : bool (default=True)
                            Decision to plot the analysis.
        save : bool (default=True)
                            Decision to save the generated tensor_values arrays.

        data_range : int (default=500)
                            Range over which the property should be evaluated. This is not applicable to the current
                            analysis as the full rdf will be calculated.
        """

        self.update_user_args(plot=plot, save=save, data_range=data_range, export=export)
        self.data_files = []
        self.savgol_order = savgol_order
        self.savgol_window_length = savgol_window_length

        out = self.run_analysis()

        self.experiment.save_class()
        # need to move save_class() to here, because it can't be done in the experiment any more!

        return out

    def _get_rdf_data(self):
        """
        Fill the data_files list with filenames of the rdf tensor_values
        """

        database = PropertiesDatabase(name=os.path.join(self.experiment.database_path, 'property_database'))

        return database.load_data({"property": "RDF"})

    def _load_rdf_from_file(self, system_property: SystemProperty):
        """
        Load the raw rdf tensor_values from a directory
        """

        radii = []
        rdf = []
        for _bin in system_property.data:
            radii.append(_bin.x)
            rdf.append(_bin.y)

        self.radii = np.array(radii)[1:].astype(float)
        self.rdf = np.array(rdf)[1:].astype(float)

    def _autocorrelation_time(self):
        """
        Not needed in this analysis
        """
        raise NotApplicableToAnalysis

    def _calculate_potential_of_mean_force(self):
        """
        Calculate the potential of mean force
        """

        self.pomf = -1 * boltzmann_constant * self.experiment.temperature * np.log(self.rdf)

    def _get_max_values(self):
        """
        Calculate the maximums of the rdf
        """
        filtered_data = apply_savgol_filter(self.pomf, order=self.savgol_order, window_length=self.savgol_window_length)

        peaks = find_peaks(filtered_data)[0]  # Find the maximums in the filtered dataset

        return [peaks[0], peaks[1]]

    def _find_minimum(self):
        """
        Find the minimum of the pomf function

        This function calls an implementation of the Golden-section search algorithm to determine the minimum of the
        potential of mean-force function.

        Returns
        -------
        pomf_indices : list
                Location of the minimums of the pomf values.
        """

        peaks = self._get_max_values()  # get the peaks of the tensor_values post-filtering

        # Calculate the radii of the minimum range
        pomf_radii = golden_section_search([self.radii, self.pomf], self.radii[peaks[1]], self.radii[peaks[0]])

        pomf_indices = list([np.where(self.radii == pomf_radii[0])[0][0],
                             np.where(self.radii == pomf_radii[1])[0][0]])

        return pomf_indices

    def _get_pomf_value(self):
        """
        Use a min-finding algorithm to calculate the potential of mean force value
        """

        self.indices = self._find_minimum()  # update the class with the minimum value indices

        # Calculate the value and error of the potential of mean-force
        pomf_value = np.mean([self.pomf[self.indices[0]], self.pomf[self.indices[1]]])
        pomf_error = np.std([self.pomf[self.indices[0]], self.pomf[self.indices[1]]]) / np.sqrt(2)

        # Update the experiment class
        properties = {"Property": self.database_group,
                      "Analysis": self.analysis_name,
                      "Subject": self.species_tuple.split('_'),
                      "data_range": self.data_range,
                      'data': [{'x': pomf_value, 'uncertainty': pomf_error}]
                      }
        self._update_properties_file(properties)

        return pomf_value, pomf_error

    def _plot_fits(self, data: list):
        """
        Plot the predicted minimum value before parsing the other tensor_values for plotting
        """
        plt.plot(self.radii, self.pomf, label=fr'{self.species_tuple}: {data[0]: 0.3E} $\pm$ {data[1]: 0.3E}')
        plt.axvspan(self.radii[self.indices[0]], self.radii[self.indices[1]], color='y', alpha=0.5, lw=0)

    def run_post_generation_analysis(self):
        """
        Calculate the potential of mean-force and perform error analysis
        """

        # fill the tensor_values array with tensor_values

        for data in self._get_rdf_data():
            self.file_to_study = data  # Set the correct tensor_values file in the class
            self.species_tuple = "_".join([subject.subject for subject in data.subjects])
            self.data_range = data.data_range
            self._load_rdf_from_file(data)  # load up the tensor_values
            log.debug(f'rdf: {self.rdf} \t radii: {self.radii}')
            self._calculate_potential_of_mean_force()  # calculate the potential of mean-force
            _data = self._get_pomf_value()  # Determine the min values of the function and update experiment

            # Update the experiment class

            if self.save:
                properties = {"Property": self.database_group,
                              "Analysis": self.analysis_name,
                              "Subject": self.species_tuple.split('_'),
                              "data_range": self.data_range,
                              'data': [{'x': x, 'y': y} for x, y in zip(self.radii, self.pomf)],
                              'information': 'full data'
                              }
                self._update_properties_file(properties)

            if self.export:
                self._export_data(name=self._build_table_name(self.species_tuple),
                                  data=self._build_pandas_dataframe(self.radii, self.pomf))

            if self.plot:
                self._plot_fits(_data)

        if self.plot:
            plt.legend()
            plt.show()

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

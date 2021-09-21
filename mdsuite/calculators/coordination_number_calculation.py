"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Class for the calculation of the coordinated numbers

Summary
-------
"""
import logging
import numpy as np
from scipy.signal import find_peaks
from mdsuite.database.calculator_database import Parameters
from mdsuite.utils.exceptions import NotApplicableToAnalysis, CannotPerformThisAnalysis
from mdsuite.calculators.calculator import Calculator, call
from mdsuite.utils.meta_functions import golden_section_search
from mdsuite.utils.meta_functions import apply_savgol_filter
from bokeh.models import BoxAnnotation

log = logging.getLogger(__name__)


class CoordinationNumbers(Calculator):
    """
    Class for the calculation of coordination numbers

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
    rdf = None : list
                        rdf tensor_values being studied.
    radii = None : list
                        radii tensor_values corresponding to the rdf.
    integral_data : list
                        integrated rdf tensor_values.
    species_tuple : list
                        A list of species combinations being studied.
    indices : list
                        A list of indices which correspond to to correct coordination numbers.

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.CoordinationNumbers(savgol_order = 2, savgol_window_length = 17)
    """

    def __init__(self, **kwargs):
        """
        Python constructor

        Parameters
        ----------
        experiment : class object
                        Class object of the experiment.
        """

        super().__init__(**kwargs)
        self.file_to_study = None
        self.rdf = None
        self.radii = None
        self.integral_data = None
        self.species_tuple = None
        self.indices = None
        self.savgol_order = None
        self.savgol_window_length = None

        self.post_generation = True

        self.database_group = 'Coordination_Numbers'
        self.x_label = r'$$\text{r}$$'
        self.y_label = 'CN'
        self.analysis_name = 'Coordination_Numbers'

    @call
    def __call__(self, plot: bool = True, save: bool = True, data_range: int = 1, export: bool = False,
                 savgol_order: int = 2, savgol_window_length: int = 17):
        """

        Parameters
        ----------
        plot : bool (default=True)
                            Decision to plot the analysis.
        save : bool (default=True)
                            Decision to save the generated tensor_values arrays.

        data_range : int (default=500)
                            Range over which the property should be evaluated.
                            This is not applicable to the current analysis as
                            the full rdf will be calculated.
        export : bool
                If true, export the data directly to a csv.
        savgol_order : int
                Order of the savgol polynomial filter
        savgol_window_length : int
                Window length of the savgol filter.

        Returns
        -------
        None.
        """

        # Calculate the rdf if it has not been done already
        if self.experiment.radial_distribution_function_state is False:
            self.experiment.run.RadialDistributionFunction(plot=False, n_batches=-1)

        self.update_user_args(plot=plot, save=save, data_range=data_range, export=export)

        self.savgol_order = savgol_order
        self.savgol_window_length = savgol_window_length

    def _get_density(self, species: str) -> float:
        """
        Use the species_tuple in front of the name to get information about the pair
        """

        species = species.split("_")  # get an array of the species being studied
        rdf_number_of_atoms = len(self.experiment.species[species[0]]['indices'])  # get the number of atoms in the RDF

        return rdf_number_of_atoms / self.experiment.volume

    def _autocorrelation_time(self):
        """
        Not needed in this analysis
        """
        raise NotApplicableToAnalysis

    def _integrate_rdf(self, density):
        """
        Integrate the rdf currently in the class state
        """

        self.integral_data = []  # empty the integral tensor_values array for analysis
        for i in range(1, len(self.radii)):  # Loop over number_of_bins in the rdf
            # Integrate the function up to the bin.
            self.integral_data.append(np.trapz((np.array(self.radii[1:i]) ** 2) * self.rdf[1:i], x=self.radii[1:i]))

        self.integral_data = np.array(self.integral_data) * 4 * np.pi * density  # Scale the result by the density

    def _get_max_values(self):
        """
        Get the max values of the rdf for use in the minimum calculation

        Returns
        -------
        peaks : list
                If an exception is not raised, the function will return a list of peaks in the rdf.
        """

        filtered_data = apply_savgol_filter(self.rdf, order=self.savgol_order, window_length=self.savgol_window_length)
        peaks = find_peaks(filtered_data, height=1.0)[0]  # get the maximum values

        # Check that more than one peak exists. If not, the GS search cannot be performed.
        if len(peaks) < 2:
            print("Not enough peaks were found for the minimum analysis (First shell). Try adjusting the filter "
                  "parameters or re-calculating the RDF for a smoother function.")
            raise CannotPerformThisAnalysis
        else:
            return [peaks[0], peaks[1], peaks[2]]  # return peaks if they exist

    def _find_minimums(self):
        """
        Use min finding algorithm to determine the minimums of the function

        Returns
        -------
        cn_indices_1, cn_indices_2 : tuple
                Returns a tuple of indices which can be evaluated on the CN function to get the correct values.
        """

        peaks = self._get_max_values()  # get the max value indices

        # Calculate the range in which the coordination numbers should exist.
        cn_radii_1 = golden_section_search([self.radii, self.rdf], self.radii[peaks[1]], self.radii[peaks[0]])
        cn_radii_2 = golden_section_search([self.radii, self.rdf], self.radii[peaks[2]], self.radii[peaks[1]])

        # Locate the indices of the radii values
        cn_indices_1 = list([np.where(self.radii == cn_radii_1[0])[0][0],
                             np.where(self.radii == cn_radii_1[1])[0][0]])
        cn_indices_2 = list([np.where(self.radii == cn_radii_2[0])[0][0],
                             np.where(self.radii == cn_radii_2[1])[0][0]])

        return cn_indices_1, cn_indices_2

    def _get_coordination_numbers(self):
        """
        Calculate the coordination numbers
        """

        self.indices = self._find_minimums()  # get the minimums

        # Calculate the coordination numbers by averaging over the two values returned by _find_minimums
        first_shell = np.mean([self.integral_data[self.indices[0][0]], self.integral_data[self.indices[0][1]]])
        first_shell_error = np.std([self.integral_data[self.indices[0][0]],
                                    self.integral_data[self.indices[0][1]]]) / np.sqrt(2)

        second_shell = np.mean([self.integral_data[self.indices[1][0]],
                                self.integral_data[self.indices[1][1]]]) - first_shell
        second_shell_error = np.std([self.integral_data[self.indices[1][0]],
                                     self.integral_data[self.indices[1][1]]]) / np.sqrt(2)


        return first_shell, first_shell_error

    def run_post_generation_analysis(self):
        """
        Calculate the coordination numbers and perform error analysis
        """
        calculations = self._get_rdf_data()
        for data in calculations:  # Loop over all existing RDFs
            log.debug(f"Computing coordination numbers for {data.subjects}")
            self.data_range = data.data_range
            self._load_rdf_from_file(data)  # load the tensor_values from it
            density = self._get_density(data.subjects[0])  # calculate the density

            self.species_tuple = "_".join(data.subjects)

            self._integrate_rdf(density)  # integrate the rdf
            self._find_minimums()  # get the minimums of the rdf being studied
            _data = self._get_coordination_numbers()  # calculate the coordination numbers and update the experiment

            properties = Parameters(
                Property=self.database_group,
                Analysis=self.analysis_name,
                data_range=self.data_range,
                data=[{'coordination_number': _data[0],
                       'uncertainty': _data[1]}],
                Subject=[self.species_tuple]
            )
            data = properties.data
            data += [{'r': x, 'cn': y} for x, y in
                     zip(self.radii[1:], self.integral_data)]
            properties.data = data
            self.update_database(properties)

            # Plot the tensor_values if required
            if self.plot:
                model_1 = BoxAnnotation(
                    left=self.radii[self.indices[0][0]],
                    right=self.radii[self.indices[0][1]],
                    fill_alpha=0.1,
                    fill_color='red')
                model_2 = BoxAnnotation(
                    left=self.radii[self.indices[1][0]],
                    right=self.radii[self.indices[1][1]],
                    fill_alpha=0.1,
                    fill_color='red')
                self.run_visualization(
                    x_data=self.radii[1:],
                    y_data=np.array(self.integral_data),
                    title=fr'{self.species_tuple}: {_data[0]: 0.3E} +- {_data[1]: 0.3E}',
                    layouts=[model_1, model_2]
                )

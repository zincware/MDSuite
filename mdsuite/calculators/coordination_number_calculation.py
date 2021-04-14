"""
Class for the calculation of the coordinated numbers

Summary
-------
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import h5py as hf

from typing import Union
# MDSuite imports
from mdsuite.utils.exceptions import *
from mdsuite.calculators.calculator import Calculator

from mdsuite.utils.meta_functions import golden_section_search
from mdsuite.utils.meta_functions import apply_savgol_filter

plt.style.use('classic')


class CoordinationNumbers(Calculator):
    """
    Class for the calculation of coordination numbers

    Attributes
    ----------
    experiment : class object
                        Class object of the experiment.
    plot : bool (default=True)
                        Decision to plot the analysis.
    save : bool (default=True)
                        Decision to save the generated tensor_values arrays.

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
    data_directory : str
                        The directory in which to find this tensor_values.
    data_files : list
                        list of files to be analyzed.
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
    """

    def __init__(self, experiment, plot: bool = True, save: bool = True, data_range: int = 1):
        """
        Python constructor

        Parameters
        ----------
        experiment : class object
                        Class object of the experiment.
        plot : bool (default=True)
                            Decision to plot the analysis.
        save : bool (default=True)
                            Decision to save the generated tensor_values arrays.

        data_range : int (default=500)
                            Range over which the property should be evaluated. This is not applicable to the current
                            analysis as the full rdf will be calculated.
        x_label : str
                            How to label the x axis of the saved plot.
        y_label : str
                            How to label the y axis of the saved plot.
        analysis_name : str
                            Name of the analysis. used in saving of the tensor_values and figure.
        """

        super().__init__(experiment, plot, save, data_range)
        self.file_to_study = None
        self.data_files = []
        self.rdf = None
        self.radii = None
        self.integral_data = None
        self.species_tuple = None
        self.indices = None

        self.post_generation = True

        self.database_group = 'coordination_numbers'
        self.x_label = r'r ($\AA$)'
        self.y_label = 'CN'
        self.analysis_name = 'Coordination_Numbers'

        # Calculate the rdf if it has not been done already
        if self.experiment.radial_distribution_function_state is False:
            self.experiment.run_computation('RadialDistributionFunction', plot=True, n_batches=-1)

    def _get_rdf_data(self):
        """
        Fill the data_files list with filenames of the rdf tensor_values
        """
        with hf.File(os.path.join(self.experiment.database_path, 'analysis_data.hdf5'), 'r') as db:
            for item in db['radial_distribution_function']:  # loop over the files
                self.data_files.append(item)  # Append to the data_file attribute

    def _get_density(self) -> float:
        """
        Use the species_tuple in front of the name to get information about the pair
        """

        species = self.species_tuple.split("_")  # get an array of the species being studied
        rdf_number_of_atoms = len(self.experiment.species[species[0]]['indices'])  # get the number of atoms in the RDF

        return rdf_number_of_atoms / self.experiment.volume

    def _load_rdf_from_file(self):
        """
        Load the raw rdf tensor_values from a directory
        """

        with hf.File(os.path.join(self.experiment.database_path, 'analysis_data.hdf5'), 'r') as db:
            self.radii, self.rdf = db['radial_distribution_function'][self.file_to_study]

    def _autocorrelation_time(self):
        """
        Not needed in this analysis
        """
        raise NotApplicableToAnalysis

    def _integrate_rdf(self):
        """
        Integrate the rdf currently in the class state
        """

        self.integral_data = []  # empty the integral tensor_values array for analysis
        for i in range(1, len(self.radii)):  # Loop over number_of_bins in the rdf
            # Integrate the function up to the bin.
            self.integral_data.append(np.trapz((np.array(self.radii[1:i]) ** 2) * self.rdf[1:i], x=self.radii[1:i]))

        density = self._get_density()  # calculate the density
        self.integral_data = np.array(self.integral_data) * 4 * np.pi * density  # Scale the result by the density

    def _get_max_values(self):
        """
        Get the max values of the rdf for use in the minimum calculation

        Returns
        -------
        peaks : list
                If an exception is not raised, the function will return a list of peaks in the rdf.
        """

        filtered_data = apply_savgol_filter(self.rdf)  # filter the tensor_values
        peaks = find_peaks(filtered_data, height=1.0)[0]  # get the maximum values

        # Check that more than one peak exists. If not, the GS search cannot be performed.
        if len(peaks) < 2:
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

        second_shell = np.mean([self.integral_data[self.indices[1][0]], self.integral_data[self.indices[1][1]]]) - \
                       first_shell
        second_shell_error = np.std([self.integral_data[self.indices[1][0]],
                                     self.integral_data[self.indices[1][1]]]) / np.sqrt(2)

        # update the experiment information file
        self._update_properties_file(item=self.species_tuple, sub_item='first_shell', add=True,
                                     data=[str(first_shell), str(first_shell_error)])
        self._update_properties_file(item=self.species_tuple, sub_item='second_shell', add=True,
                                     data=[str(second_shell), str(second_shell_error)])

    def _plot_coordination_shells(self):
        """
        Plot the calculated coordination numbers on top of the rdfs
        """

        fig, ax1 = plt.subplots()  # define the plot
        ax1.plot(self.radii, self.rdf, label=f"{self.species_tuple} RDF")  # plot the RDF
        ax1.set_ylabel('RDF')  # set the y_axis label on the LHS
        ax2 = ax1.twinx()  # split the axis
        ax2.set_ylabel('CN')  # set the RHS y axis label
        # plot the CN as a continuous function
        ax2.plot(self.radii[1:], np.array(self.integral_data), 'r')  # , markersize=1, label=f"{self.species_tuple} CN")
        # Plot the first and second shell values as a small window.
        ax1.axvspan(self.radii[self.indices[0][0]] - 0.01, self.radii[self.indices[0][1]] + 0.01, color='g')
        ax1.axvspan(self.radii[self.indices[1][0]] - 0.01, self.radii[self.indices[1][1]] + 0.01, color='b')
        ax1.set_xlabel(r'r ($\AA$)')  # set the x-axis label
        plt.savefig(f'{self.species_tuple}.svg', dpi=800)
        plt.show()

    def run_post_generation_analysis(self):
        """
        Calculate the coordination numbers and perform error analysis
        """

        self._get_rdf_data()  # fill the tensor_values array with tensor_values
        for data in self.data_files:  # Loop over all existing RDFs
            self.file_to_study = data  # set the working file
            self.species_tuple = data[:-29]  # set the tuple
            self._load_rdf_from_file()  # load the tensor_values from it
            self._integrate_rdf()  # integrate the rdf
            self._find_minimums()  # get the minimums of the rdf being studied
            self._get_coordination_numbers()  # calculate the coordination numbers and update the experiment class

            # Save the tensor_values if required
            if self.save:
                self._save_data(f"{self.species_tuple}_{self.analysis_name}", [np.array(self.radii[1:]),
                                                                               np.array(self.integral_data)])

            # Plot the tensor_values if required
            if self.plot:
                self._plot_coordination_shells()
            self._plot_data(manual=True)  # Call plot function to prevent later issues

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

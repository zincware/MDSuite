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

import numpy as np
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import h5py as hf
from typing import Union

# MDSuite imports
from mdsuite.utils.exceptions import *
from mdsuite.calculators.calculator import Calculator

from mdsuite.utils.meta_functions import golden_section_search
from mdsuite.utils.meta_functions import apply_savgol_filter
from mdsuite.utils.units import boltzmann_constant


class PotentialOfMeanForce(Calculator):
    """
    Class for the calculation of the potential of mean-force

    The potential of mean-force is a measure of the binding strength between atomic species in a experiment. Mathematically
    one may write

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
    species_tuple : list
                        A list of species combinations being studied.
    pomf : list
                        List of tensor_values of the potential of mean-force for the current analysis.
    """

    def __init__(self, experiment, plot=True, save=True, data_range=1):
        """
        Python constructor for the class

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
        self.file_to_study = None                                             # RDF file being studied
        self.data_files = []                                                  # array of the files in tensor_values directory
        self.rdf = None                                                       # rdf being studied
        self.radii = None                                                     # radii of the rdf
        self.species_tuple = None                                             # Which species are being studied
        self.pomf = None                                                      # potential of mean force array
        self.indices = None                                                   # Indices of the pomf range
        self.database_group = 'potential_of_mean_force_values'                # Which database_path group to save the tensor_values in
        self.x_label = r'r ($\AA$)'
        self.y_label = r'$w^{(2)}(r)$'
        self.analysis_name = 'Potential_of_Mean_Force'

        self.post_generation = True

    def _get_rdf_data(self):
        """
        Fill the data_files list with filenames of the rdf tensor_values
        """
        with hf.File(os.path.join(self.experiment.database_path, 'analysis_data.hdf5'), 'r') as db:
            for item in db['radial_distribution_function']:  # loop over the files
                self.data_files.append(item)  # Append to the data_file attribute

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

    def _calculate_potential_of_mean_force(self):
        """
        Calculate the potential of mean force
        """

        self.pomf = -1 * boltzmann_constant * self.experiment.temperature * np.log(self.rdf)

    def _get_max_values(self):
        """
        Calculate the maximums of the rdf
        """
        filtered_data = apply_savgol_filter(self.pomf)  # Apply a filter to the tensor_values to smooth curve
        peaks = find_peaks(filtered_data)[0]               # Find the maximums in the filtered dataset

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
        pomf_error = np.std([self.pomf[self.indices[0]], self.pomf[self.indices[1]]])/np.sqrt(2)

        # Update the experiment class
        self._update_properties_file(item=self.species_tuple, data= [str(pomf_value), str(pomf_error)])

    def _plot_fits(self):
        """
        Plot the predicted minimum value before parsing the other tensor_values for plotting
        """
        plt.plot(self.radii, self.pomf, label=f'{self.species_tuple}')
        plt.axvspan(self.radii[self.indices[0]], self.radii[self.indices[1]], color='y', alpha=0.5, lw=0)

    def run_post_generation_analysis(self):
        """
        Calculate the potential of mean-force and perform error analysis
        """

        self._get_rdf_data()  # fill the tensor_values array with tensor_values

        for data in self.data_files:
            self.file_to_study = data                  # Set the correct tensor_values file in the class
            self.species_tuple = data[:-33]            # set the tuple
            self._load_rdf_from_file()                 # load up the tensor_values
            self._calculate_potential_of_mean_force()  # calculate the potential of mean-force
            self._get_pomf_value()                     # Determine the min values of the function and update experiment

            # Plot and save the tensor_values if necessary
            if self.save:
                self._save_data(f"{self.species_tuple}_{self.analysis_name}", [self.radii, self.pomf])

            if self.plot:
                self._plot_fits()

        if self.plot:
            self._plot_data()

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
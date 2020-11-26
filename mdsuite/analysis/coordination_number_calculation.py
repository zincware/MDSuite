""" Class for the calculation of the coordinated numbers """

# Standard imports
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# MDSuite imports
from mdsuite.utils.exceptions import *
from mdsuite.analysis.analysis import Analysis

from mdsuite.utils.meta_functions import golden_section_search
from mdsuite.utils.meta_functions import apply_savgol_filter

class _CoordinationNumbers(Analysis):
    """ Class for the calculation of coordination numbers

    """

    def __init__(self, obj, plot=True, save=True, data_range=None, x_label=r'r ($\AA$)', y_label='CN',
                 analysis_name='Coordination_Numbers'):
        """ Python constructor """
        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name)
        self.file_to_study = None  # Which rdf to use in the analysis at a time
        self.data_directory = f'{obj.storage_path}/{obj.analysis_name}/data'
        self.data_files = []  # array of the files in data directory
        self.rdf = None  # rdf being studied
        self.radii = None  # radii of the rdf
        self.integral_data = None  # integrated rdf being studied
        self.species_tuple = None  # Which species are being studied - important for the density calculation
        self.indices = None  # indices of the coordination shells being studied

        # Calculate the rdf if it has not been done already
        if self.parent.radial_distribution_function_state is False:
            self.parent.radial_distribution_function()

    def _get_rdf_data(self):
        """ Fill the data_files list with filenames of the rdf data """
        files = os.listdir(self.data_directory)  # load the directory contents
        for item in files:
            if item[-32:] == 'radial_distribution_function.npy':
                self.data_files.append(item)

    def _get_density(self):
        """ Use the species_tuple in front of the name to get information about the pair """

        species = self.species_tuple.split("_")  # get an array of the species being studied

        rdf_number_of_atoms = len(self.parent.species[species[0]]['indices'])

        return rdf_number_of_atoms/self.parent.volume

    def _load_rdf_from_file(self):
        """ Load the raw rdf data from a directory """

        self.radii, self.rdf = np.load(f'{self.data_directory}/{self.file_to_study}', allow_pickle=True)

    def _autocorrelation_time(self):
        """ Not needed in this analysis """
        raise NotApplicableToAnalysis

    def _integrate_rdf(self):
        """ Integrate the rdf currently in the class state """

        self.integral_data = []  # empty the integral data array for analysis
        for i in range(1, len(self.radii)):
            self.integral_data.append(np.trapz((np.array(self.radii[1:i])**2)*self.rdf[1:i], x=self.radii[1:i]))

        density = self._get_density()  # calculate the density

        self.integral_data = np.array(self.integral_data)*4*np.pi*density

    def _get_max_values(self):
        """ get the max values of the rdf for use in the minimum calculation """

        filtered_data = apply_savgol_filter(self.rdf)  # filter the data
        peaks = find_peaks(filtered_data, height=1.0)[0]  # get the maximum values

        if len(peaks) < 2:
            raise CannotPerformThisAnalysis
        else:
            return [peaks[0], peaks[1], peaks[2]]

    def _find_minimums(self):
        """ Use min finding algorithm to determine the minimums of the function """

        peaks = self._get_max_values()  # get the max value indices

        # Calculate the range in which the coordination numbers should exist.
        cn_radii_1 = golden_section_search([self.radii, self.rdf], self.radii[peaks[1]], self.radii[peaks[0]])
        cn_radii_2 = golden_section_search([self.radii, self.rdf], self.radii[peaks[2]], self.radii[peaks[1]])

        cn_indices_1 = list([np.where(self.radii == cn_radii_1[0])[0][0],
                        np.where(self.radii == cn_radii_1[1])[0][0]])
        cn_indices_2 = list([np.where(self.radii == cn_radii_2[0])[0][0],
                        np.where(self.radii == cn_radii_2[1])[0][0]])

        return cn_indices_1, cn_indices_2

    def _get_coordination_numbers(self):
        """ Calculate the coordination numbers """

        self.indices = self._find_minimums()  # get the minimums

        first_shell = np.mean([self.integral_data[self.indices[0][0]], self.integral_data[self.indices[0][1]]])
        first_shell_error = np.std([self.integral_data[self.indices[0][0]],
                                   self.integral_data[self.indices[0][1]]])/np.sqrt(2)

        second_shell = np.mean([self.integral_data[self.indices[1][0]], self.integral_data[self.indices[1][1]]])
        second_shell_error = np.std([self.integral_data[self.indices[1][0]],
                                    self.integral_data[self.indices[1][1]]])/np.sqrt(2)

        self.parent.coordination_numbers[self.species_tuple] = {'first_shell': [first_shell, first_shell_error],
                                                                'second_shell': [second_shell, second_shell_error]}


    def _plot_coordination_shells(self):
        """ Plot the calculated coordination numbers on top of the rdfs """

        plt.plot(self.radii, self.rdf, label=f"{self.species_tuple} RDF")
        plt.plot(self.radii[self.indices[0]], self.rdf[self.indices[0]], 'o')
        plt.plot(self.radii[self.indices[1]], self.rdf[self.indices[1]], 'x')


    def run_analysis(self):
        """ Calculate the coordination numbers and perform error analysis """

        self._get_rdf_data()  # fill the data array with data

        for data in self.data_files:
            self.file_to_study = data  # set the working file
            self.species_tuple = data[:-33]  # set the tuple
            self._load_rdf_from_file()  # load the data from it
            self._integrate_rdf()  # integrate the rdf
            self._find_minimums()  # get the minimums of the rdf being studied
            self._get_coordination_numbers()  # calculate the coordination numbers and update the experiment class

            if self.save:
                self._save_data(f"{self.species_tuple}_{self.analysis_name}", [self.radii, self.integral_data])

            if self.plot:
                plt.plot(self.radii[1:], np.array(self.integral_data), label=f"{self.species_tuple} CCN")
                self._plot_coordination_shells()

        if self.plot:
            self._plot_data()

""" Class for the calculation of the coordinated numbers """

import numpy as np
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# MDSuite imports
from mdsuite.utils.constants import *
from mdsuite.utils.exceptions import *
from mdsuite.analysis.analysis import Analysis

from mdsuite.utils.meta_functions import golden_section_search
from mdsuite.utils.meta_functions import apply_savgol_filter


class StructureFactor(Analysis):
    """ Class for the calculation of the structure factor


    Attributes
    ----------
    obj : class object
                        Class object of the experiment.
    plot : bool (default=True)
                        Decision to plot the analysis.
    save : bool (default=True)
                        Decision to save the generated data arrays.

    data_range : int (default=500)
                        Range over which the property should be evaluated. This is not applicable to the current
                        analysis as the full rdf will be calculated.
    x_label : str
                        How to label the x axis of the saved plot.
    y_label : str
                        How to label the y axis of the saved plot.
    analysis_name : str
                        Name of the analysis. used in saving of the data and figure.
    file_to_study : str
                        The data file corresponding to the rdf being studied.
    data_directory : str
                        The directory in which to find this data.
    data_files : list
                        list of files to be analyzed.
    rdf = None : list
                        rdf data being studied.
    radii = None : list
                        radii data corresponding to the rdf.
    species_tuple : list
                        A list of species combinations being studied.
    pomf : list
                        List of data of the potential of mean-force for the current analysis.
    """

    def __init__(self, obj, plot=True, save=True, data_range=None, x_label=r'r ($\AA$)', y_label=r'$w^{(2)}(r)$',
                 analysis_name='Structure_Factor'):
        """ Python constructor for the class

        Parameters
        ----------
        obj : class object
                        Class object of the experiment.
        plot : bool (default=True)
                            Decision to plot the analysis.
        save : bool (default=True)
                            Decision to save the generated data arrays.

        data_range : int (default=500)
                            Range over which the property should be evaluated. This is not applicable to the current
                            analysis as the full rdf will be calculated.
        x_label : str
                            How to label the x axis of the saved plot.
        y_label : str
                            How to label the y axis of the saved plot.
        analysis_name : str
                            Name of the analysis. used in saving of the data and figure.
        """

        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name)
        self.file_to_study = None                                             # RDF file being studied
        self.data_directory = f'{obj.storage_path}/{obj.analysis_name}/data'  # directory in which data is stored
        self.data_files = []                                                  # array of the files in data directory
        self.rdf = None                                                       # rdf being studied
        self.radii = None                                                     # radii of the rdf
        self.species_tuple = None                                             # Which species are being studied
        self.pomf = None                                                      # potential of mean force array
        self.indices = None                                                   # Indices of the pomf range

    def _get_rdf_data(self):
        """ Fill the data_files list with filenames of the rdf data """
        files = os.listdir(self.data_directory)  # load the directory contents
        for item in files:
            if item[-32:] == 'radial_distribution_function.npy':
                self.data_files.append(item)

    def _load_rdf_from_file(self):
        """ Load the raw rdf data from a directory """

        self.radii, self.rdf = np.load(f'{self.data_directory}/{self.file_to_study}', allow_pickle=True)

    def _autocorrelation_time(self):
        """ Not needed in this analysis """
        raise NotApplicableToAnalysis

    def partial_structure_factor(self):
        print(' \n hello \n')



    def _plot_fits(self):
        """ Plot the predicted minimum value before parsing the other data for plotting """
        plt.plot(self.radii, self.pomf, label=f'{self.species_tuple}')
        plt.axvspan(self.radii[self.indices[0]], self.radii[self.indices[1]], color='y', alpha=0.5, lw=0)

    def run_analysis(self):
        """ Calculate the potential of mean-force and perform error analysis """

        self._get_rdf_data()  # fill the data array with data

        for data in self.data_files:
            self.file_to_study = data                  # Set the correct data file in the class
            self.species_tuple = data[:-33]            # set the tuple
            self._load_rdf_from_file()                 # load up the data
            self._calculate_potential_of_mean_force()  # calculate the potential of mean-force
            self._get_pomf_value()                     # Determine the min values of the function and update experiment

            # Plot and save the data if necessary
            if self.save:
                self._save_data(f"{self.species_tuple}_{self.analysis_name}", [self.radii, self.pomf])

            if self.plot:
                self._plot_fits()

        if self.plot:
            self._plot_data()


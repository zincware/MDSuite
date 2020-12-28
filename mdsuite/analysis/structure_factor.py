""" Class for the calculation of the coordinated numbers """

import numpy as np
import os
import pandas as pd
from scipy.integrate import simps
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

# MDSuite imports
from mdsuite.utils.constants import *
from mdsuite.utils.exceptions import *
from mdsuite.analysis.analysis import Analysis

from mdsuite.utils.meta_functions import golden_section_search
from mdsuite.utils.meta_functions import apply_savgol_filter

from mdsuite import data as static_data
from importlib.resources import open_text

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

    def __init__(self, obj, Q, rho, plot=True, save=True, data_range=None, x_label=r'r ($\AA$)', y_label=r'$w^{(2)}(r)$',
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
        self.Q = Q
        self.rho = rho
        with open_text(static_data, 'form_fac_coeffs.csv') as file:
            self.coeff_atomic_formfactor = pd.read_csv(file, sep=',')


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

    def _plot_fits(self):
        """ Plot the predicted minimum value before parsing the other data for plotting """
        plt.plot(self.radii, self.pomf, label=f'{self.species_tuple}')
        plt.axvspan(self.radii[self.indices[0]], self.radii[self.indices[1]], color='y', alpha=0.5, lw=0)

    def atomic_form_factors(self):
        species = self.file_to_study.split('_')[0:2]
        print(species[0])
        print(list(self.coeff_atomic_formfactor.columns))
        print('stuff', self.coeff_atomic_formfactor.loc[self.coeff_atomic_formfactor['Element'] == 'H1-'].iloc[0, 0])
        return 1


    def weight_factors(self, c_1, c_2):
        f_factors = self.atomic_form_factors()
        weight = c_1 * c_2
        return weight


    def partial_structure_factor(self):
        test = False
        if test:
            self.radii=np.linspace(0,4*np.pi)
            integrand=np.sin(self.radii)
            running_integral = cumtrapz(integrand, self.radii, initial=0.0)
            integral = simps(integrand, self.radii)
            S = 1 + 4 * np.pi * self.rho * integral
            print('integral ', integral)
            print('s ', S)
            plt.figure()
            plt.plot(self.radii, running_integral)
            plt.show()



        integrand = np.zeros(len(self.radii))
        #print(self.rdf)
        for counter,radius in enumerate(self.radii):
            if np.isnan(self.rdf[counter]) :
                self.rdf[counter]= 0
            if radius == 0:
                integrand[counter] = 0
                continue
            integrand[counter] = radius**2 * np.sin(self.Q*radius)/(self.Q*radius) * (self.rdf[counter] - 1)
        #print(integrand)
        running_integral = cumtrapz(integrand, self.radii, initial=0.0)
        integral = simps(integrand, self.radii)
        S_12 = 1 + 4 * np.pi * self.rho * integral
        print('integral ', integral)
        print('s ', S_12)
        plt.figure()
        plt.plot(self.radii, running_integral, label='integral')
        plt.plot(self.radii, self.rdf, label='rdf')
        plt.legend()
        plt.show()
        return S_12



    def run_analysis(self):
        """ Calculate the potential of mean-force and perform error analysis """

        self._get_rdf_data()
        print(self.data_files)
        self.file_to_study = self.data_files[1]
        self._load_rdf_from_file()
        #print(self.radii)
        S_12 = self.partial_structure_factor()
        weight = self.weight_factors(c_1=0.5, c_2=0.5)
        S_in = weight * S_12
        print('S_in', S_in)

        #self._get_rdf_data()  # fill the data array with data

        # for data in self.data_files:
        #     self.file_to_study = data                  # Set the correct data file in the class
        #     self.species_tuple = data[:-33]            # set the tuple
        #     self._load_rdf_from_file()                 # load up the data
        #     self._calculate_potential_of_mean_force()  # calculate the potential of mean-force
        #     self._get_pomf_value()                     # Determine the min values of the function and update experiment
        #
        #     # Plot and save the data if necessary
        #     if self.save:
        #         self._save_data(f"{self.species_tuple}_{self.analysis_name}", [self.radii, self.pomf])
        #
        #     if self.plot:
        #         self._plot_fits()
        #
        # if self.plot:
        #     self._plot_data()

""" Class for the calculation of the total structure factor for X-rays using the Faber-Ziman formalism"""

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
    """ Class for the calculation of the total structure factor for X-ray scattering
        using the Faber-Ziman partial structure factors. This analysis is valid for a magnitude of the X-ray
        scattering vector Q < 25 * 1/Angstrom
        Explicitly equations 9, 10 and 11 of the paper
        'DFT Accurate Interatomic Potential for Molten NaCl from MachineLearning' from
        Samuel Tovey, Anand Narayanan Krishnamoorthy, Ganesh Sivaraman, Jicheng Guo,
        Chris Benmore,Andreas Heuer, and Christian Holm are implemented.

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
    """

    def __init__(self, obj,  plot=True, save=True, data_range=None, x_label=r'Q ($\AA ^{-1}$)', y_label=r'S(Q)',
                 analysis_name='total_structure_factor'):
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
        self.Q_arr = np.linspace(0.5, 25, 700)                 # array of scattering vectors
        self.obj = obj                                         # instance of the experiment class
        self.rho = self.obj.number_of_atoms / (self.obj.box_array[0] *
                                               self.obj.box_array[1] * self.obj.box_array[2])  # particle density
        with open_text(static_data, 'form_fac_coeffs.csv') as file:
            self.coeff_atomic_formfactor = pd.read_csv(file, sep=',')  # stores coefficients for atomic form factors

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

    def gauss(self, a, b, scattering_scalar):
        """ Calculates the gauss functions that are required for the atomic form factors """
        return a * np.exp(-b * (scattering_scalar / (4 * np.pi))**2)

    def atomic_form_factors(self, scattering_scalar):
        """ Calculates the atomic form factors for all elements in the species
        dictionary and returns it """
        atomic_form_di = {}
        for el in self.obj.species:
            if self.obj.species[el]['charge'][0] == 0:
                el_key = el
            if self.obj.species[el]['charge'][0] > 0:
                el_key = el + str(self.obj.species[el]['charge'][0]) + '+'
            if self.obj.species[el]['charge'][0] < 0:
                el_key = el + str(self.obj.species[el]['charge'][0])[1:] + '-'
            el_frame =self.coeff_atomic_formfactor.loc[self.coeff_atomic_formfactor['Element'] == el_key]
            atomic_form_fac = self.gauss(el_frame.iloc[0,1], el_frame.iloc[0,2], scattering_scalar) + self.gauss(el_frame.iloc[0,3], el_frame.iloc[0,4], scattering_scalar)\
                            + self.gauss(el_frame.iloc[0,5], el_frame.iloc[0,6], scattering_scalar) + self.gauss(el_frame.iloc[0,7], el_frame.iloc[0,8], scattering_scalar) + el_frame.iloc[0,9]
            atomic_form_di[el] = {}
            atomic_form_di[el]['atomic_form_factor'] = atomic_form_fac
        return atomic_form_di

    def molar_fractions(self):
        """ Calculates the molar fractions for all elements in the species dictionary and add it to the species
        dictionary """
        for el in self.obj.species:
            self.obj.species[el]['molar_fraction'] = len(self.obj.species[el]['indices'])/self.obj.number_of_atoms

    def species_densities(self):
        """ Calculates the particle densities for all the speies in the species dictionary and add it to the species
		dictionary """
        for el in self.obj.species:
            self.obj.species[el]['particle_density'] = len(self.obj.species[el]['indices']) / (self.obj.box_array[0] *
                                                                                               self.obj.box_array[1] * self.obj.box_array[2])

    def average_atomic_form_factor(self, scattering_scalar):
        """ Calculates the average atomic form factor """
        sum1 = 0
        atomic_form_facs = self.atomic_form_factors(scattering_scalar)
        for el in self.obj.species:
            sum1 += self.obj.species[el]['molar_fraction'] * atomic_form_facs[el]['atomic_form_factor']
        average_atomic_factor = sum1**2
        return average_atomic_factor

    def partial_structure_factor(self, scattering_scalar, elements):
        """ Calculates the partial structure factor """
        integrand = np.zeros(len(self.radii))
        for counter, radius in enumerate(self.radii):
            if np.isnan(self.rdf[counter]):
                self.rdf[counter] = 0
            if radius == 0:
                integrand[counter] = 0
                continue
            integrand[counter] = radius**2 * np.sin(scattering_scalar * radius)/(scattering_scalar * radius) * (self.rdf[counter] - 1)
        running_integral = cumtrapz(integrand, self.radii, initial=0.0)
        integral = simps(integrand, self.radii)
        particle_density = self.obj.species[elements[0]]['particle_density']        #given g_ab take the particle density of a
        s_12 = 1 + 4 * np.pi * particle_density * integral
        return s_12, running_integral, integral

    def weight_factor(self, scattering_scalar):
        """ Calculates the weight factor """
        species_lst = self.file_to_study.split('_')[:2]
        c_a = self.obj.species[species_lst[0]]['molar_fraction']
        c_b = self.obj.species[species_lst[1]]['molar_fraction']
        form_factors = self.atomic_form_factors(scattering_scalar)
        avg_form_fac = self.average_atomic_form_factor(scattering_scalar)
        atom_form_fac_a = form_factors[species_lst[0]]['atomic_form_factor']
        atom_form_fac_b = form_factors[species_lst[1]]['atomic_form_factor']
        weight = c_a * c_b * atom_form_fac_a * atom_form_fac_b / avg_form_fac
        return weight

    def total_structure_factor(self, scattering_scalar):
        """ Calculates the total structure factor by summing the products of weight_factor * partial_structure_factor """
        self.atomic_form_factors(scattering_scalar)
        total_struc_fac = 0
        for filename in self.data_files:
            self.file_to_study = filename
            self._load_rdf_from_file()
            elements = self.file_to_study.split('_')[:2]     # get the names of the species for the current rdf
            s_12, _, _ = self.partial_structure_factor(scattering_scalar, elements)
            s_in = self.weight_factor(scattering_scalar) * s_12
            if elements[0] != elements[1]:                  # S_ab and S_ba need to be considered
                elements2 = [elements[1], elements[0]]
                s_21, _, _ = self.partial_structure_factor(scattering_scalar, elements2)
                s_in2 = self.weight_factor(scattering_scalar) * s_21
                s_in += s_in2
            total_struc_fac += s_in
        return total_struc_fac

    def run_analysis(self):
        """ Calculates the total structure factor for all the different Q-values of the Q_arr (magnitude of the scattering vector) """
        test = False
        if test:
            self.run_test()
        else:
            self._get_rdf_data()
            self.molar_fractions()
            self.species_densities()
            total_structure_factor_li =[]
            for counter, scattering_scalar in enumerate(self.Q_arr):
                total_structure_factor_li.append(self.total_structure_factor(scattering_scalar))

            total_structure_factor_li = np.array(total_structure_factor_li)
            if self.plot:
                plt.plot(self.Q_arr, total_structure_factor_li, 'bo', label='total structure factor')

            # Save a figure if required
            if self.plot:
                self._plot_data()

            if self.save:
                self._save_data(f"{self.analysis_name}", [self.Q_arr, total_structure_factor_li])

    def run_test(self):
        """ A function that can be used to test the correctness of the structure_factor class """
        print('\nStarting sturucture factor test \n')

        self._get_rdf_data()
        self.molar_fractions()
        self.species_densities()
        print('rho ', self.rho)
        #print(self.obj.species)
        form_facs = self.atomic_form_factors(10)
        print('Q: 10,', ' form factors: ', form_facs)
        avg_atom_f_factor = self.average_atomic_form_factor(10)
        print('Q: 10', 'average atomic form factor: ', avg_atom_f_factor)

        scattering_scalar =0.5
        self.file_to_study = self.data_files[2]
        print(self.file_to_study)
        self._load_rdf_from_file()
        s_12, running_integral, integral = self.partial_structure_factor(scattering_scalar)
        print('partial structure factor', s_12)
        print('integral: ', integral)
        print('weight', self.weight_factor(scattering_scalar))
        plt.figure()
        plt.plot(self.radii, self.rdf, label='rdf')
        plt.plot(self.radii, running_integral, label='running integral')
        plt.legend()
        plt.show()
        #
        # self.build_atomic_form_dict()
        # self.Q_counter = 0
        # self.Q = 5.1
        # print(self.atom_form_facs)
        # self.total_structure_factor()
        # print(self.total_struc_facs)



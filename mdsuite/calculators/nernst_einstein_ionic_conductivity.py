"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.
"""

""" Calculate the Nernst-Einstein Conductivity of a experiment """

import numpy as np
import os
import yaml
import sys
import operator
from typing import Union
from mdsuite.calculators.calculator import Calculator
from mdsuite.utils.units import boltzmann_constant, elementary_charge


class NernstEinsteinIonicConductivity(Calculator):
    """
    Class for the calculation of the Nernst-Einstein ionic conductivity
    """

    def __init__(self, experiment, corrected: bool = False, plot: bool = False, data_range: int = 1,
                 export: bool = False, species: list = None):
        """
        Standard constructor

        Parameters
        ----------
        experiment : Experiment
                Experiment class from which to read
        corrected : bool
                If true, the corrected Nernst Einstein will also be calculated
        """
        super().__init__(experiment, plot=plot, save=False, data_range=data_range, export=export)
        self.corrected = corrected
        self.post_generation = True
        self.data = self._load_data()  # tensor_values to be read in
        self.truth_table = self._build_truth_table()  # build truth table for analysis

        self.database_group = "Ionic_Conductivity"
        if species is None:
            self.species = list(self.experiment.species)
        else:
            self.species = species

    def _load_data(self):
        """
        Load tensor_values from a yaml file

        Returns
        -------
        tensor_values: dict
                A dictionary of tensor_values stored in the yaml file
        """

        test = self.experiment.export_property_data({'Property': 'Diffusion_Coefficients'})
        return test

    def _build_truth_table(self):
        """
        Builds a truth table to communicate which tensor_values is available to the analysis.

        Returns
        -------
        truth_table : list
                A truth table communication which tensor_values is available for the analysis.
        """

        case_1 = self.experiment.export_property_data({'Property': 'Diffusion_Coefficients',
                                                       "Analysis": "Green_Kubo_Self_Diffusion_Coefficients"})
        case_2 = self.experiment.export_property_data({'Property': 'Diffusion_Coefficients',
                                                       "Analysis": "Green_Kubo_Distinct_Diffusion_Coefficients"})
        case_3 = self.experiment.export_property_data({'Property': 'Diffusion_Coefficients',
                                                       "Analysis": "Einstein_Self_Diffusion_Coefficients"})
        case_4 = self.experiment.export_property_data({'Property': 'Diffusion_Coefficients',
                                                       "Analysis": "Einstein_Distinct_Diffusion_Coefficients"})
        truth_table = [list(map(operator.not_, [not case_1, not case_2])),
                       list(map(operator.not_, [not case_3, not case_4]))]
        return truth_table

    def _nernst_einstein(self, diffusion_information: list):
        """
        Calculate the Nernst-Einstein ionic conductivity

        Parameters
        ----------
        diffusion_information : list
                A list of dictionaries loaded from the SQL properties database.

        Returns
        -------
        Nernst-Einstein Ionic conductivity of the experiment in units of S/cm
        """

        # evaluate the prefactor
        numerator = self.experiment.number_of_atoms * (elementary_charge ** 2)
        denominator = boltzmann_constant * self.experiment.temperature * \
                      (self.experiment.volume * (self.experiment.units['length'] ** 3))
        prefactor = numerator / denominator

        conductivity = 0.0
        uncertainty = 0.0
        for item in diffusion_information:
            diffusion_coefficient = item['data']
            diffusion_uncertainty = item['uncertainty']
            species = item['Subject']
            charge_term = self.experiment.species[species]['charge'][0] ** 2
            mass_fraction_term = len(self.experiment.species[species]['indices']) / self.experiment.number_of_atoms
            conductivity += diffusion_coefficient * charge_term * mass_fraction_term
            uncertainty += diffusion_uncertainty * charge_term * mass_fraction_term
            data_range = item['data_range']

        return [prefactor * conductivity, prefactor * uncertainty, data_range]

    def _corrected_nernst_einstein(self, self_diffusion_information: list, distinct_diffusion_information: list):
        """
        Calculate the corrected Nernst-Einstein ionic conductivity

        Parameters
        ----------
        self_diffusion_information : dict
                dictionary containing information about self diffusion
        distinct_diffusion_information : dict
                dictionary containing information about distinct diffusion

        Returns
        -------
        Corrected Nernst-Einstein ionic conductivity in units of S/cm
        """

        # evaluate the prefactor
        numerator = self.experiment.number_of_atoms * (elementary_charge ** 2)
        denominator = boltzmann_constant * self.experiment.temperature * \
                      (self.experiment.volume * (self.experiment.units['length'] ** 3))
        prefactor = numerator / denominator

        conductivity = 0.0
        uncertainty = 0.0
        for item in self_diffusion_information:
            diffusion_coefficient = item['data']
            diffusion_uncertainty = item['uncertainty']
            species = item['Subject']
            charge_term = self.experiment.species[species]['charge'][0] ** 2
            mass_fraction_term = len(self.experiment.species[species]['indices']) / self.experiment.number_of_atoms
            conductivity += diffusion_coefficient * charge_term * mass_fraction_term
            uncertainty += diffusion_uncertainty * charge_term * mass_fraction_term
            data_range = item['data_range']
        for item in distinct_diffusion_information:
            diffusion_coefficient = item['data']
            diffusion_uncertainty = item['uncertainty']
            constituents = item['Subject'].split("_")
            charge_term = self.experiment.species[constituents[0]]['charge'][0] * \
                         self.experiment.species[constituents[1]]['charge'][0]
            mass_fraction_term = (len(
                self.experiment.species[constituents[0]]['indices']) / self.experiment.number_of_atoms) * \
                                 (len(self.experiment.species[constituents[1]][
                                          'indices']) / self.experiment.number_of_atoms)
            conductivity += diffusion_coefficient * charge_term * mass_fraction_term
            uncertainty += diffusion_uncertainty * charge_term * mass_fraction_term

        return [prefactor * conductivity, prefactor * uncertainty, data_range]

    def _run_nernst_einstein(self):
        """
        Process truth table and run all possible nernst-einstein calculations

        Returns
        -------

        """
        ne_table = [self.truth_table[0][0], self.truth_table[1][0]]
        if ne_table[0]:
            input_data = [self.experiment.export_property_data({'Property': 'Diffusion_Coefficients',
                                                                "Analysis": "Green_Kubo_Self_Diffusion_Coefficients",
                                                                "Subject": species})[0] for species in self.species]
            data = self._nernst_einstein(input_data)
            properties = {"Property": self.database_group,
                          "Analysis": "Green_Kubo_Nernst_Einstein_Ionic_Conductivity",
                          "Subject": f"System",
                          "data_range": data[2],
                          'data': data[0],
                          'uncertainty': data[1]}
            self._update_properties_file(properties)

        if ne_table[1]:
            input_data = [self.experiment.export_property_data({'Property': 'Diffusion_Coefficients',
                                                                "Analysis": "Einstein_Self_Diffusion_Coefficients",
                                                                "Subject": species})[0] for species in self.species]
            data = self._nernst_einstein(input_data)
            properties = {"Property": self.database_group,
                          "Analysis": "Einstein_Nernst_Einstein_Ionic_Conductivity",
                          "Subject": "System",
                          "data_range": data[2],
                          "data": data[0],
                          "uncertainty": data[1]}
            self._update_properties_file(properties)

        if not any(ne_table):
            print("There is no values to analyse, please run a diffusion calculation to proceed")
            sys.exit(1)

    def _run_corrected_nernst_einstein(self):
        """
        Process truth table and run all possible nernst-einstein calculations

        Returns
        -------
        Updates the experiment database_path
        """

        cne_table = [self.truth_table[0][1], self.truth_table[1][1]]

        if cne_table[0]:
            input_self = [self.experiment.export_property_data({'Property': 'Diffusion_Coefficients',
                                                                "Analysis": "Green_Kubo_Self_Diffusion_Coefficients",
                                                                "Subject": species})[0] for species in self.species]
            input_distinct = [self.experiment.export_property_data({'Property': 'Diffusion_Coefficients',
                                                                "Analysis": "Green_Kubo_Distinct_Diffusion_Coefficients",
                                                                "Subject": species})[0] for species in self.species]
            data = self._corrected_nernst_einstein(input_self, input_distinct)
            properties = {"Property": self.database_group,
                          "Analysis": "Green_Kubo_Corrected_Nernst_Einstein_Ionic_Conductivity",
                          "Subject": f"System",
                          "data_range": data[2],
                          'data': data[0],
                          'uncertainty': data[1]}
            self._update_properties_file(properties)

        if cne_table[1]:
            input_self = [self.experiment.export_property_data({'Property': 'Diffusion_Coefficients',
                                                                "Analysis": "Einstein_Self_Diffusion_Coefficients",
                                                                "Subject": species})[0] for species in self.species]
            input_distinct = [self.experiment.export_property_data({'Property': 'Diffusion_Coefficients',
                                                                "Analysis": "Einstein_Distinct_Diffusion_Coefficients",
                                                                "Subject": species})[0] for species in self.species]
            data = self._corrected_nernst_einstein(input_self, input_distinct)
            properties = {"Property": self.database_group,
                          "Analysis": "Einstein_Corrected_Nernst_Einstein_Ionic_Conductivity",
                          "Subject": "System",
                          "data_range": data[2],
                          "data": data[0],
                          "uncertainty": data[1]}
            self._update_properties_file(properties)

    def run_post_generation_analysis(self):
        """
        Run the analysis
        """

        self._run_nernst_einstein()
        if self.corrected:
            self._run_corrected_nernst_einstein()

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

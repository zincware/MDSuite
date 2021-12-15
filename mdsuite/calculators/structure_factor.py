"""
MDSuite: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
"""
import logging
from dataclasses import dataclass
from importlib.resources import open_text

import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz, simps
from tqdm import tqdm

from mdsuite import data
from mdsuite.calculators.calculator import Calculator

log = logging.getLogger(__name__)


@dataclass
class Args:
    """
    Data class for the saved properties.
    """

    data_range: int
    correlation_time: int
    atom_selection: np.s_
    tau_values: np.s_
    molecules: bool
    species: list


class StructureFactor(Calculator):
    """
    Class for the calculation of the total structure factor for X-ray scattering
    using the Faber-Ziman partial structure factors. This analysis is valid for a
    magnitude of the X-ray scattering vector Q < 25 * 1/Angstrom
    Explicitly equations 9, 10 and 11 of the paper

    'DFT Accurate Interatomic Potential for Molten NaCl from MachineLearning' from
    Samuel Tovey, Anand Narayanan Krishnamoorthy, Ganesh Sivaraman, Jicheng Guo,
    Chris Benmore,Andreas Heuer, and Christian Holm are implemented.

    Attributes
    ----------
    experiment : class object
                        Class object of the experiment.
    x_label : str
                        How to label the x axis of the saved plot.
    y_label : str
                        How to label the y axis of the saved plot.
    analysis_name : str
                        Name of the analysis. used in saving of the tensor_values and
                        figure.
    file_to_study : str
                        The tensor_values file corresponding to the rdf being studied.
    data_files : list
                        list of files to be analyzed.
    rdf = None : list
                        rdf tensor_values being studied.
    radii = None : list
                        radii tensor_values corresponding to the rdf.

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.StructureFactor()

    Notes
    -----
    In order to use the structure factor calculator both the masses and the
    charges of each species must be present. If they are not correct, the structure
    factor will not work.
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
        self.Q_arr = np.linspace(0.5, 25, 700)

        self.post_generation = True

        self.x_label = r"$$\text{Q} / nm ^{-1}$$"
        self.y_label = r"$$\text{S(Q)}$$"
        self.analysis_name = "total_structure_factor"

        self.rho = None

        with open_text(data, "form_fac_coeffs.csv") as file:
            self.coeff_atomic_formfactor = pd.read_csv(
                file, sep=","
            )  # stores coefficients for atomic form factors

    def __call__(self, plot=True, data_range=1):
        """
        Parameters
        ----------
        plot : bool (default=True)
                            Decision to plot the analysis.
        data_range : int (default=500)
                            Range over which the property should be evaluated.
                            This is not applicable to the current analysis as
                            the full rdf will be calculated.

        Returns
        -------
        None.
        """

        out = {}
        for experiment in self.experiments:
            self.experiment = experiment

            self.plot = plot

            self.rho = self.experiment.number_of_atoms / (
                self.experiment.box_array[0]
                * self.experiment.box_array[1]
                * self.experiment.box_array[2]
            )

            if self.load_data:
                out[self.experiment.name] = self.experiment.export_property_data(
                    {"Analysis": self.analysis_name}
                )
            else:
                out[self.experiment.name] = self.run_analysis()

        if len(self.experiments) > 1:
            return out
        else:
            return out[self.experiment.name]

    @staticmethod
    def gauss(a, b, scattering_scalar):
        """
        Calculates the gauss functions that are required for the atomic form factors
        """
        return a * np.exp(-b * (scattering_scalar / (4 * np.pi)) ** 2)

    def atomic_form_factors(self, scattering_scalar):
        """
        Calculates the atomic form factors for all elements in the species
        dictionary and returns it
        """
        atomic_form_di = {}
        for el in list(self.experiment.species):
            if self.experiment.species[el]["charge"][0] == 0:
                el_key = el
            elif self.experiment.species[el]["charge"][0] > 0:
                el_key = el + str(self.experiment.species[el]["charge"][0]) + "+"
            elif self.experiment.species[el]["charge"][0] < 0:
                el_key = el + str(self.experiment.species[el]["charge"][0])[1:] + "-"
            else:
                print("Impossible input")
                return
            el_frame = self.coeff_atomic_formfactor.loc[
                self.coeff_atomic_formfactor["Element"] == el_key
            ]
            atomic_form_fac = (
                self.gauss(el_frame.iloc[0, 1], el_frame.iloc[0, 2], scattering_scalar)
                + self.gauss(el_frame.iloc[0, 3], el_frame.iloc[0, 4], scattering_scalar)
                + self.gauss(el_frame.iloc[0, 5], el_frame.iloc[0, 6], scattering_scalar)
                + self.gauss(el_frame.iloc[0, 7], el_frame.iloc[0, 8], scattering_scalar)
                + el_frame.iloc[0, 9]
            )
            atomic_form_di[el] = {}
            atomic_form_di[el]["atomic_form_factor"] = atomic_form_fac
        return atomic_form_di

    @property
    def molar_fractions(self) -> dict:
        """
        Calculates the molar fractions for all elements in the species
        dictionary and add it to the species dictionary

        # TODO value is not cached!
        """
        molar_fractions = {}

        for el in self.experiment.species:
            molar_fractions[el] = (
                self.experiment.species[el].n_particles / self.experiment.number_of_atoms
            )

        return molar_fractions

    @property
    def species_densities(self):
        """Calculates the particle densities

        Calculates the particle densities for all the species in the species
        dictionary and add it to the species dictionary

        # TODO this is uncached!!
        # TODO was species[x]["species_densities"] ever used somewhere?
        #   this needs tests!
        """
        species_densities = {}
        for el in self.experiment.species:
            species_densities[el] = self.experiment.species[el].n_particles / (
                self.experiment.box_array[0]
                * self.experiment.box_array[1]
                * self.experiment.box_array[2]
            )
        return species_densities

    def average_atomic_form_factor(self, scattering_scalar):
        """
        Calculates the average atomic form factor
        """
        sum1 = 0
        atomic_form_facs = self.atomic_form_factors(scattering_scalar)
        for el in self.experiment.species:
            sum1 += self.molar_fractions[el] * atomic_form_facs[el]["atomic_form_factor"]
        average_atomic_factor = sum1 ** 2
        return average_atomic_factor

    def partial_structure_factor(self, scattering_scalar, elements):
        """
        Calculates the partial structure factor
        """
        integrand = np.zeros(len(self.radii))
        for counter, radius in enumerate(self.radii):
            if np.isnan(self.rdf[counter]):
                self.rdf[counter] = 0
            if radius == 0:
                integrand[counter] = 0
                continue
            integrand[counter] = (
                radius ** 2
                * np.sin(scattering_scalar * radius)
                / (scattering_scalar * radius)
                * (self.rdf[counter] - 1)
            )
        running_integral = cumtrapz(integrand, self.radii, initial=0.0)
        integral = simps(integrand, self.radii)
        particle_density = self.experiment.species[elements[0]][
            "particle_density"
        ]  # given g_ab take the particle density of a
        s_12 = 1 + 4 * np.pi * particle_density * integral
        return s_12, running_integral, integral

    def weight_factor(self, scattering_scalar, species_lst):
        """
        Calculates the weight factor
        """
        c_a = self.molar_fractions[species_lst[0]]
        c_b = self.molar_fractions[species_lst[1]]
        form_factors = self.atomic_form_factors(scattering_scalar)
        avg_form_fac = self.average_atomic_form_factor(scattering_scalar)
        atom_form_fac_a = form_factors[species_lst[0]]["atomic_form_factor"]
        atom_form_fac_b = form_factors[species_lst[1]]["atomic_form_factor"]
        weight = c_a * c_b * atom_form_fac_a * atom_form_fac_b / avg_form_fac
        return weight

    def total_structure_factor(self, scattering_scalar):
        """
        Calculates the total structure factor by summing the products of
        weight_factor * partial_structure_factor
        """
        self.atomic_form_factors(scattering_scalar)
        total_struc_fac = 0
        for data_ in self._get_rdf_data():
            log.debug(f"Loaded data: {data_}")
            self._load_rdf_from_file(data_)
            log.debug(f"Loaded RDF: {self.rdf.shape} and radii: {self.radii.shape}")
            elements = data_.subjects
            log.debug(f"Elements are: {elements}")
            s_12, _, _ = self.partial_structure_factor(scattering_scalar, elements)
            s_in = self.weight_factor(scattering_scalar, elements) * s_12
            if elements[0] != elements[1]:  # S_ab and S_ba need to be considered
                elements2 = [elements[1], elements[0]]
                s_21, _, _ = self.partial_structure_factor(scattering_scalar, elements2)
                s_in2 = self.weight_factor(scattering_scalar, elements) * s_21
                s_in += s_in2
            total_struc_fac += s_in
        return total_struc_fac

    def run_calculator(self):
        """
        Calculates the total structure factor for all the different Q-values
        of the Q_arr (magnitude of the scattering vector)
        """
        self._get_rdf_data()
        total_structure_factor_li = []
        for counter, scattering_scalar in tqdm(
            enumerate(self.Q_arr),
            total=len(self.Q_arr),
            desc="Structure factor calculation",
        ):
            total_structure_factor_li.append(
                self.total_structure_factor(scattering_scalar)
            )
        total_structure_factor_li = np.array(total_structure_factor_li)

        data = {"q": self.Q_arr.tolist(), "s(q)": total_structure_factor_li.tolist()}

        self.queue_data(data=data, subjects=["System"])

        if self.plot:
            self.run_visualization(
                x_data=self.Q_arr,
                y_data=total_structure_factor_li,
                title="total structure factor",
            )

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
import numpy as np
from scipy.signal import find_peaks
from mdsuite.utils.exceptions import CannotPerformThisAnalysis
from mdsuite.calculators.calculator import Calculator, call
from mdsuite.utils.meta_functions import golden_section_search
from mdsuite.utils.meta_functions import apply_savgol_filter
from bokeh.models import BoxAnnotation
from dataclasses import dataclass


log = logging.getLogger(__name__)


@dataclass
class Args:
    """
    Data class for the saved properties.
    """

    savgol_order: int
    savgol_window_length: int


class CoordinationNumbers(Calculator):
    """
    Class for the calculation of coordination numbers

    Attributes
    ----------
    experiment : class object
                        Class object of the experiment.

    data_range : int (default=500)
                        Range over which the property should be evaluated. This is not
                        applicable to the current analysis as the full rdf will be
                        calculated.
    x_label : str
                        How to label the x axis of the saved plot.
    y_label : str
                        How to label the y axis of the saved plot.
    analysis_name : str
                        Name of the analysis. used in saving of the tensor_values and
                        figure.
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
                        A list of indices which correspond to to correct coordination
                        numbers.

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.CoordinationNumbers(savgol_order = 2,
                                                   savgol_window_length = 17)
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
        self.data_range = 1

        self.post_generation = True

        self.database_group = "Coordination_Numbers"
        self.x_label = r"$$\text{r} /  nm$$"
        self.y_label = "CN"
        self.analysis_name = "Coordination_Numbers"
        self.result_keys = [
            "coordination_number",
            "uncertainty",
            "left1",
            "right1",
            "left2",
            "right2",
        ]
        self.result_series_keys = ["r", "cn"]

    @call
    def __call__(
        self,
        plot: bool = True,
        data_range: int = 1,
        savgol_order: int = 2,
        savgol_window_length: int = 17,
    ):
        """

        Parameters
        ----------
        plot : bool (default=True)
                            Decision to plot the analysis.
        data_range : int (default=500)
                            Range over which the property should be evaluated.
                            This is not applicable to the current analysis as
                            the full rdf will be calculated.
        savgol_order : int
                Order of the savgol polynomial filter
        savgol_window_length : int
                Window length of the savgol filter.

        Returns
        -------
        None.
        """
        self.plot = plot

        # set args that will affect the computation result
        self.args = Args(
            savgol_order=savgol_order,
            savgol_window_length=savgol_window_length,
        )

    def _get_density(self, species: str) -> float:
        """
        Use the species_tuple in front of the name to get information about the pair
        """

        species = species.split("_")  # get an array of the species being studied
        rdf_number_of_atoms = len(
            self.experiment.species[species[0]]["indices"]
        )  # get the number of atoms in the RDF

        return rdf_number_of_atoms / self.experiment.volume

    def _integrate_rdf(self, density):
        """
        Integrate the rdf currently in the class state
        """

        self.integral_data = []  # empty the integral tensor_values array for analysis
        for i in range(1, len(self.radii)):  # Loop over number_of_bins in the rdf
            # Integrate the function up to the bin.
            self.integral_data.append(
                np.trapz(
                    (np.array(self.radii[1:i]) ** 2) * self.rdf[1:i], x=self.radii[1:i]
                )
            )

        self.integral_data = (
            np.array(self.integral_data) * 4 * np.pi * density
        )  # Scale the result by the density

    def _get_max_values(self):
        """
        Get the max values of the rdf for use in the minimum calculation

        Returns
        -------
        peaks : list
                If an exception is not raised, the function will return a list of peaks
                in the rdf.
        """

        filtered_data = apply_savgol_filter(
            self.rdf,
            order=self.args.savgol_order,
            window_length=self.args.savgol_window_length,
        )
        peaks = find_peaks(filtered_data, height=1.0)[0]  # get the maximum values

        # Check that more than one peak exists. If not, the GS search cannot be
        # performed.
        if len(peaks) < 2:
            print(
                "Not enough peaks were found for the minimum analysis (First shell)."
                " Try adjusting the filter parameters or re-calculating the RDF for a"
                " smoother function."
            )
            raise CannotPerformThisAnalysis
        else:
            return [peaks[0], peaks[1], peaks[2]]  # return peaks if they exist

    def _find_minimums(self):
        """
        Use min finding algorithm to determine the minimums of the function

        Returns
        -------
        cn_indices_1, cn_indices_2 : tuple
                Returns a tuple of indices which can be evaluated on the CN function to
                get the correct values.
        """

        peaks = self._get_max_values()  # get the max value indices

        # Calculate the range in which the coordination numbers should exist.
        cn_radii_1 = golden_section_search(
            [self.radii, self.rdf], self.radii[peaks[1]], self.radii[peaks[0]]
        )
        cn_radii_2 = golden_section_search(
            [self.radii, self.rdf], self.radii[peaks[2]], self.radii[peaks[1]]
        )

        # Locate the indices of the radii values
        cn_indices_1 = list(
            [
                np.where(self.radii == cn_radii_1[0])[0][0],
                np.where(self.radii == cn_radii_1[1])[0][0],
            ]
        )
        cn_indices_2 = list(
            [
                np.where(self.radii == cn_radii_2[0])[0][0],
                np.where(self.radii == cn_radii_2[1])[0][0],
            ]
        )

        return cn_indices_1, cn_indices_2

    def _get_coordination_numbers(self):
        """
        Calculate the coordination numbers
        """

        self.indices = self._find_minimums()  # get the minimums

        # Calculate the coordination numbers by averaging over the two values
        # returned by _find_minimums
        first_shell = np.mean(
            [
                self.integral_data[self.indices[0][0]],
                self.integral_data[self.indices[0][1]],
            ]
        )
        first_shell_error = (
            np.std(
                [
                    self.integral_data[self.indices[0][0]],
                    self.integral_data[self.indices[0][1]],
                ]
            )
            / np.sqrt(2)
        )

        # # TODO what about second shell?!
        # second_shell = (
        #     np.mean(
        #         [
        #             self.integral_data[self.indices[1][0]],
        #             self.integral_data[self.indices[1][1]],
        #         ]
        #     )
        #     - first_shell
        # )
        # second_shell_error = (
        #     np.std(
        #         [
        #             self.integral_data[self.indices[1][0]],
        #             self.integral_data[self.indices[1][1]],
        #         ]
        #     )
        #     / np.sqrt(2)
        # )

        return first_shell, first_shell_error

    def run_calculator(self):
        """
        Calculate the coordination numbers and perform error analysis
        """
        calculations = self.experiment.run.RadialDistributionFunction(plot=False)
        self.data_range = calculations.data_range
        for (
            selected_species,
            vals,
        ) in calculations.data_dict.items():  # Loop over all existing RDFs
            log.debug(f"Computing coordination numbers for {selected_species}")
            self.radii = np.array(vals["x"]).astype(float)[1:]
            self.rdf = np.array(vals["y"]).astype(float)[1:]
            self.selected_species = selected_species.split("_")
            self.species_tuple = selected_species  # depreciated

            density = self._get_density(
                self.selected_species[0]
            )  # calculate the density

            self._integrate_rdf(density)  # integrate the rdf
            self._find_minimums()  # get the minimums of the rdf being studied
            _data = (
                self._get_coordination_numbers()
            )  # calculate the coordination numbers and update the experiment

            data = {
                self.result_keys[0]: _data[0],
                self.result_keys[1]: _data[1],
                self.result_keys[2]: self.radii[self.indices[0][0]],
                self.result_keys[3]: self.radii[self.indices[0][1]],
                self.result_keys[4]: self.radii[self.indices[1][0]],
                self.result_keys[5]: self.radii[self.indices[1][1]],
                self.result_series_keys[0]: self.radii[1:].tolist(),
                self.result_series_keys[1]: self.integral_data.tolist(),
            }

            self.queue_data(data=data, subjects=self.selected_species)

    def plot_data(self, data):
        """Plot the CN"""
        # Plot the tensor_values if required
        for selected_species, val in data.items():
            model_1 = BoxAnnotation(
                left=val[self.result_keys[2]],
                right=val[self.result_keys[3]],
                fill_alpha=0.1,
                fill_color="red",
            )
            model_2 = BoxAnnotation(
                left=val[self.result_keys[4]],
                right=val[self.result_keys[5]],
                fill_alpha=0.1,
                fill_color="red",
            )
            self.run_visualization(
                x_data=val[self.result_series_keys[0]],
                y_data=val[self.result_series_keys[1]],
                title=(
                    fr"{selected_species}: {val[self.result_keys[0]]: 0.3E} +-"
                    fr" {val[self.result_keys[1]]: 0.3E}"
                ),
                layouts=[model_1, model_2],
            )

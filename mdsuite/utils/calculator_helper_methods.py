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
Static methods used in calculators are kept here rather than polluting the parent class.
"""
import random

import numpy as np
from scipy.optimize import curve_fit


def fit_einstein_curve(data: list) -> list:
    """
    Fit operation for Einstein calculations

    Parameters
    ----------
    data : list
            x and y tensor_values for the fitting [np.array, np.array] of
            (2, data_range)

    Returns
    -------
    fit results : list
            A tuple list with the fit value along with the error of the fit
    """

    fits = []  # define an empty fit array so errors may be extracted

    def func(x, m, a):
        """
        Standard linear function for fitting.

        Parameters
        ----------
        x : list/np.array
                x axis tensor_values for the fit
        m : float
                gradient of the line
        a : float
                scalar offset, also the y-intercept for those who did not
                get much maths in school.

        Returns
        -------

        """
        return m * x + a

    # get the logarithmic dataset
    log_y = np.log10(data[1][1:])
    log_x = np.log10(data[0][1:])

    min_end_index, max_end_index = int(0.8 * len(log_y)), int(len(log_y) - 1)
    min_start_index, max_start_index = int(0.3 * len(log_y)), int(0.5 * len(log_y))

    for _ in range(100):
        end_index = random.randint(min_end_index, max_end_index)
        start_index = random.randint(min_start_index, max_start_index)

        popt, pcov = curve_fit(
            func, log_x[start_index:end_index], log_y[start_index:end_index]
        )
        fits.append(10 ** popt[1])

    return [np.mean(fits), np.std(fits)]


# def _optimize_einstein_data_range(self, data: np.array):
#     """
#     Optimize the tensor_values range of a experiment using the Einstein
#     method of calculation.
#
#     Parameters
#     ----------
#     data : np.array (2, data_range)
#             MSD to study
#
#     Returns
#     -------
#     Updates the data_range attribute of the class state
#
#     Notes
#     -----
#     TODO: Update this and add it to the code.
#     """
#
#     def func(x, m, a):
#         """
#         Standard linear function for fitting.
#
#         Parameters
#         ----------
#         x : list/np.array
#                 x axis tensor_values for the fit
#         m : float
#                 gradient of the line
#         a : float
#                 scalar offset, also the y-intercept for those who did not
#                 get much maths in school.
#
#         Returns
#         -------
#
#         """
#
#         return m * x + a
#
#     # get the logarithmic dataset
#     log_y = np.log10(data[0])
#     log_x = np.log10(data[1])
#
#     end_index = int(len(log_y) - 1)
#     start_index = int(0.4 * len(log_y))
#
#     popt, pcov = curve_fit(
#         func, log_x[start_index:end_index], log_y[start_index:end_index]
#     )
#
#     if 0.85 < popt[0] < 1.15:
#         self.loop_condition = True
#
#     else:
#         try:
#             self.args.data_range = int(1.1 * self.args.data_range)
#             self.time = np.linspace(
#                 0.0,
#                 self.args.data_range
#                 * self.experiment.time_step
#                 * self.experiment.sample_rate,
#                 self.args.data_range,
#             )
#
#             # end the calculation if the tensor_values range exceeds the relevant
#             # bounds
#             if (
#                 self.args.data_range
#                 > self.experiment.number_of_configurations
#                 - self.args.correlation_time
#             ):
#                 print("Trajectory not long enough to perform analysis.")
#                 raise RangeExceeded
#         except RangeExceeded:
#             raise RangeExceeded

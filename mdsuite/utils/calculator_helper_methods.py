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
import logging
from typing import Any, Iterable, Tuple, Union

import numpy as np
from numpy import ndarray
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

log = logging.getLogger(__name__)


def fit_einstein_curve(
    x_data: np.ndarray, y_data: np.ndarray
) -> Tuple[Union[ndarray, Iterable, int, float], Any]:
    """
    Fit operation for Einstein calculations

    Parameters
    ----------
    x_data : np.ndarray
            x data to use in the fitting.
    y_data : np.ndarray
            y_data to use in the fitting.

    Returns
    -------
    popt : list
            List of fit values
    pcov : list
            Covariance matrix of the fit values.
    """

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
        m * x + a
        """
        return m * x + a

    spline_data = UnivariateSpline(x_data, y_data, s=0, k=4)

    derivatives = spline_data.derivative(n=2)(x_data)

    derivatives[abs(derivatives) < 1e-5] = 0
    start_index = np.argmin(abs(derivatives))

    popt, pcov = curve_fit(func, xdata=x_data[start_index:], ydata=y_data[start_index:])

    return popt, pcov


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
#                 log.error("Trajectory not long enough to perform analysis.")
#                 raise RangeExceeded
#         except RangeExceeded:
#             raise RangeExceeded

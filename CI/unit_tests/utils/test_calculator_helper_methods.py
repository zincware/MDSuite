"""
MDSuite: A zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the zincwarecode Project.

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
Module for testing the calculator helper methods.
"""
import numpy as np
import pytest

from mdsuite.utils.calculator_helper_methods import fit_einstein_curve


class TestCalculatorHelperMethods:
    """
    Test suite for the calculator helper methods.
    """

    def test_fit_einstein_curve(self):
        """
        Test the fitting of a line.

        Notes
        -----
        Tests the following:

        * Returns correct gradient on a straight line
        * Returns correct gradient on a multi-regime line
        """
        x_data = np.linspace(0, 1000, 1000)

        # y = 5x + c
        y_data = 5 * x_data + 3

        popt, pcov, gradients, gradient_errors = fit_einstein_curve(
            x_data=x_data, y_data=y_data, fit_max_index=999
        )
        assert popt[0] == pytest.approx(5.0, 0.01)

        # exp(0.05x)x^2 + 5x + 3
        y_data = np.exp(-0.05 * x_data) * x_data**2 + 5 * x_data + 3

        popt, pcov, gradients, gradient_errors = fit_einstein_curve(
            x_data=x_data, y_data=y_data, fit_max_index=999
        )
        assert popt[0] == pytest.approx(5.0, 0.01)

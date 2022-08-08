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

from mdsuite.calculators.calculator import Calculator

log = logging.getLogger(__name__)


class NernstEinsteinIonicConductivity(Calculator):
    """
    Class for the calculation of the Nernst-Einstein ionic conductivity

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.NernstEinsteinIonicConductivity()

    """

    def __init__(
        self,
        corrected: bool = False,
        plot: bool = False,
        data_range: int = 1,
        export: bool = False,
        species: list = None,
        save: bool = True,
        **kwargs
    ):
        """
        Standard constructor

        Parameters
        ----------
        corrected : bool
                If true, correct the output with the distinct diffusion
                coefficient.
        export : bool
                If true, generate a csv file after the analysis.
        plot : bool
                if true, plot the output.
        species : list
                List of species on which to operate.
        data_range : int
                Data range to use in the analysis.
        save : bool
                if true, save the output.
        """
        super().__init__(**kwargs)
        self.post_generation = True

        # Properties
        self._truth_table = None

        self.database_group = "Ionic_Conductivity"
        self.analysis_name = "Nernst_Einstein_Ionic_Conductivity"

        raise NotImplementedError("This calculator is not yet ready.")

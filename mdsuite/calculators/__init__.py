"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

__init__ module for the analysis directory
"""
from __future__ import annotations

from .calculator import Calculator
from .angular_distribution_function import AngularDistributionFunction
from .coordination_number_calculation import CoordinationNumbers
from .einstein_diffusion_coefficients import EinsteinDiffusionCoefficients
from .einstein_distinct_diffusion_coefficients import EinsteinDistinctDiffusionCoefficients
from .einstein_helfand_ionic_conductivity import EinsteinHelfandIonicConductivity
from .einstein_helfand_thermal_conductivity import EinsteinHelfandThermalConductivity
from .einstein_helfand_thermal_kinaci import EinsteinHelfandThermalKinaci
from .flux_viscosity import GreenKuboViscosityFlux
from .green_kubo_distinct_diffusion_coefficients import GreenKuboDistinctDiffusionCoefficients
from .green_kubo_ionic_conductivity import GreenKuboIonicConductivity
from .green_kubo_self_diffusion_coefficients import GreenKuboSelfDiffusionCoefficients
from .green_kubo_thermal_conductivity import GreenKuboThermalConductivity
from .green_kubo_viscosity import GreenKuboViscosity
from .kirkwood_buff_integrals import KirkwoodBuffIntegral
from .nernst_einstein_ionic_conductivity import NernstEinsteinIonicConductivity
from .potential_of_mean_force import PotentialOfMeanForce
from .radial_distribution_function import RadialDistributionFunction
from .structure_factor import StructureFactor

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from mdsuite.experiment import Experiment

__all__ = ['Calculator', 'AngularDistributionFunction', 'CoordinationNumbers', 'EinsteinDiffusionCoefficients',
           'EinsteinDistinctDiffusionCoefficients', 'EinsteinHelfandIonicConductivity',
           'EinsteinHelfandThermalConductivity', 'EinsteinHelfandThermalKinaci', 'GreenKuboViscosityFlux',
           'GreenKuboDistinctDiffusionCoefficients', 'GreenKuboIonicConductivity', 'GreenKuboSelfDiffusionCoefficients',
           'GreenKuboThermalConductivity', 'GreenKuboViscosity', 'KirkwoodBuffIntegral',
           'NernstEinsteinIonicConductivity', 'PotentialOfMeanForce', 'RadialDistributionFunction', 'StructureFactor']


class RunComputation:
    """Collection of all calculators that can be used by an experiment"""

    def __init__(self, experiment: Experiment = None, experiments: List[Experiment] = None, load_data: bool = False):
        """Collection of all calculators

        Parameters
        ----------
        experiment: Experiment
            Experiment to run the computations for
        """
        self.experiment = experiment
        self.experiments = experiments

        self.kwargs = {'experiment': experiment, 'experiments': experiments, 'load_data': load_data}

    @property
    def AngularDistributionFunction(self):
        """Calculator Property"""
        # TODO consider passing **self.<placeholder> so you can update it more easily
        return AngularDistributionFunction(experiment=self.experiment)

    @property
    def CoordinationNumbers(self):
        """Calculator Property"""
        return CoordinationNumbers(experiment=self.experiment)

    @property
    def EinsteinDiffusionCoefficients(self):
        """Calculator Property"""
        return EinsteinDiffusionCoefficients(**self.kwargs)

    @property
    def EinsteinDistinctDiffusionCoefficients(self):
        """Calculator Property"""
        return EinsteinDistinctDiffusionCoefficients(experiment=self.experiment)

    @property
    def EinsteinHelfandIonicConductivity(self):
        """Calculator Property"""
        return EinsteinHelfandIonicConductivity(experiment=self.experiment)

    @property
    def EinsteinHelfandThermalKinaci(self):
        """Calculator Property"""
        return EinsteinHelfandThermalKinaci(experiment=self.experiment)

    @property
    def GreenKuboViscosityFlux(self):
        """Calculator Property"""
        return GreenKuboViscosityFlux(experiment=self.experiment)

    @property
    def GreenKuboDistinctDiffusionCoefficients(self):
        """Calculator Property"""
        return GreenKuboDistinctDiffusionCoefficients(experiment=self.experiment)

    @property
    def GreenKuboIonicConductivity(self):
        """Calculator Property"""
        return GreenKuboIonicConductivity(experiment=self.experiment)

    @property
    def GreenKuboSelfDiffusionCoefficients(self):
        """Calculator Property"""
        return GreenKuboSelfDiffusionCoefficients(experiment=self.experiment)

    @property
    def GreenKuboThermalConductivity(self):
        """Calculator Property"""
        return GreenKuboThermalConductivity(experiment=self.experiment)

    @property
    def GreenKuboViscosity(self):
        """Calculator Property"""
        return GreenKuboViscosity(experiment=self.experiment)

    @property
    def KirkwoodBuffIntegral(self):
        """Calculator Property"""
        return KirkwoodBuffIntegral(experiment=self.experiment)

    @property
    def NernstEinsteinIonicConductivity(self):
        """Calculator Property"""
        return NernstEinsteinIonicConductivity(experiment=self.experiment)

    @property
    def PotentialOfMeanForce(self):
        """Calculator Property"""
        return PotentialOfMeanForce(experiment=self.experiment)

    @property
    def RadialDistributionFunction(self):
        """Calculator Property"""
        return RadialDistributionFunction(**self.kwargs)

    @property
    def StructureFactor(self):
        """Calculator Property"""
        return StructureFactor(experiment=self.experiment)

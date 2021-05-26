"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Module to contain the structured dict for analysis.

Summary
-------
"""
from mdsuite.calculators.angular_distribution_function import AngularDistributionFunction
from mdsuite.calculators.coordination_number_calculation import CoordinationNumbers
from mdsuite.calculators.einstein_diffusion_coefficients import EinsteinDiffusionCoefficients
from mdsuite.calculators.einstein_helfand_ionic_conductivity import EinsteinHelfandIonicConductivity
from mdsuite.calculators.einstein_helfand_thermal_conductivity import EinsteinHelfandThermalConductivity
from mdsuite.calculators.einstein_helfand_thermal_kinaci import EinsteinHelfandThermalKinaci
from mdsuite.calculators.flux_viscosity import GreenKuboViscosityFlux
from mdsuite.calculators.green_kubo_ionic_conductivity import GreenKuboIonicConductivity
from mdsuite.calculators.green_kubo_self_diffusion_coefficients import GreenKuboSelfDiffusionCoefficients
from mdsuite.calculators.green_kubo_thermal_conductivity import GreenKuboThermalConductivity
from mdsuite.calculators.green_kubo_viscosity import GreenKuboViscosity
from mdsuite.calculators.kirkwood_buff_integrals import KirkwoodBuffIntegral
from mdsuite.calculators.nernst_einstein_ionic_conductivity import NernstEinsteinIonicConductivity
from mdsuite.calculators.potential_of_mean_force import PotentialOfMeanForce
from mdsuite.calculators.radial_distribution_function import RadialDistributionFunction
from mdsuite.calculators.structure_factor import StructureFactor
from mdsuite.calculators.green_kubo_distinct_diffusion_coefficients import GreenKuboDistinctDiffusionCoefficients
from mdsuite.calculators.einstein_distinct_diffusion_coefficients import EinsteinDistinctDiffusionCoefficients

switcher_transformations = {
    'Translational_Dipole_Moment': 'TranslationalDipoleMoment',
    'Ionic_Current': 'IonicCurrent',
    'Integrated_Heat_Current': 'IntegratedHeatCurrent',
    'Thermal_Flux': 'ThermalFlux',
    'Momentum_Flux': 'MomentumFlux',
    'Kinaci_Heat_Current': 'KinaciIntegratedHeatCurrent'
}

dict_classes_db = {
    'self_diffusion_coefficients': {'Einstein_Self_Diffusion_Coefficients': {},
                                    'Green_Kubo_Self_Diffusion': {}},
    'distinct_diffusion_coefficients': {'Einstein_Distinct_Diffusion_Coefficients': {},
                                        'Green_Kubo_Distinct_Diffusion_Coefficients': {}},
    'ionic_conductivity': {},
    'thermal_conductivity': {},
    'coordination_numbers': {'Coordination_Numbers': {}},
    'potential_of_mean_force_values': {'Potential_of_Mean_Force': {}},
    'radial_distribution_function': {},
    'kirkwood_buff_integral': {},
    'structure_factor': {},
    'viscosity': {}
}

dict_classes_computations = {
    'EinsteinDiffusionCoefficients': EinsteinDiffusionCoefficients,
    'EinsteinDistinctDiffusionCoefficients': EinsteinDistinctDiffusionCoefficients,
    'GreenKuboDiffusionCoefficients': GreenKuboSelfDiffusionCoefficients,
    'GreenKuboDistinctDiffusionCoefficients': GreenKuboDistinctDiffusionCoefficients,
    'GreenKuboIonicConductivity': GreenKuboIonicConductivity,
    'EinsteinHelfandIonicConductivity': EinsteinHelfandIonicConductivity,
    'RadialDistributionFunction': RadialDistributionFunction,
    'CoordinationNumbers': CoordinationNumbers,
    'PotentialOfMeanForce': PotentialOfMeanForce,
    'KirkwoodBuffIntegral': KirkwoodBuffIntegral,
    'GreenKuboThermalConductivity': GreenKuboThermalConductivity,
    'StructureFactor': StructureFactor,
    'EinsteinHelfandThermalConductivity': EinsteinHelfandThermalConductivity,
    'GreenKuboViscosityFlux': GreenKuboViscosityFlux,
    'GreenKuboViscosity': GreenKuboViscosity,
    'AngularDistributionFunction': AngularDistributionFunction,
    'NernstEinsteinIonicConductivity': NernstEinsteinIonicConductivity,
    'EinsteinHelfandThermalKinaci': EinsteinHelfandThermalKinaci,
}

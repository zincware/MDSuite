"""
Module to contain the structured dict for analysis.

Summary
-------
"""

from mdsuite.calculators.angular_distribution_function import AngularDistributionFunction
from mdsuite.calculators.coordination_number_calculation import CoordinationNumbers
from mdsuite.calculators.einstein_diffusion_coefficients import EinsteinDiffusionCoefficients
from mdsuite.calculators.einstein_helfand_ionic_conductivity import EinsteinHelfandIonicConductivity
from mdsuite.calculators.einstein_helfand_thermal_conductivity import EinsteinHelfandThermalConductivity
from mdsuite.calculators.flux_thermal import GreenKuboThermalConductivityFlux
from mdsuite.calculators.flux_viscosity import GreenKuboViscosityFlux
from mdsuite.calculators.green_kubo_self_diffusion_coefficients import GreenKuboSelfDiffusionCoefficients
from mdsuite.calculators.green_kubo_ionic_conductivity import GreenKuboIonicConductivity
from mdsuite.calculators.green_kubo_thermal_conductivity import GreenKuboThermalConductivity
from mdsuite.calculators.green_kubo_viscosity import GreenKuboViscosity
from mdsuite.calculators.kirkwood_buff_integrals import KirkwoodBuffIntegral
from mdsuite.calculators.nernst_einstein_ionic_conductivity import NernstEinsteinIonicConductivity
from mdsuite.calculators.potential_of_mean_force import PotentialOfMeanForce
from mdsuite.calculators.radial_distribution_function import RadialDistributionFunction
from mdsuite.calculators.structure_factor import StructureFactor
from mdsuite.calculators.einstein_helfand_thermal_kinaci import EinsteinHelfandThermalKinaci

dict_classes_computations = {
    'EinsteinDiffusionCoefficients': EinsteinDiffusionCoefficients,
    'GreenKuboSelfDiffusionCoefficients': GreenKuboSelfDiffusionCoefficients,
    'GreenKuboIonicConductivity': GreenKuboIonicConductivity,
    'EinsteinHelfandIonicConductivity': EinsteinHelfandIonicConductivity,
    'RadialDistributionFunction': RadialDistributionFunction,
    'CoordinationNumbers': CoordinationNumbers,
    'PotentialOfMeanForce': PotentialOfMeanForce,
    'KirkwoodBuffIntegral': KirkwoodBuffIntegral,
    'GreenKuboThermalConductivity': GreenKuboThermalConductivity,
    'GreenKuboThermalConductivityFlux': GreenKuboThermalConductivityFlux,
    'StructureFactor': StructureFactor,
    'EinsteinHelfandThermalConductivity': EinsteinHelfandThermalConductivity,
    'GreenKuboViscosityFlux': GreenKuboViscosityFlux,
    'GreenKuboViscosity': GreenKuboViscosity,
    'AngularDistributionFunction': AngularDistributionFunction,
    'NernstEinsteinIonicConductivity': NernstEinsteinIonicConductivity,
    'EinsteinHelfandThermalKinaci': EinsteinHelfandThermalKinaci
}

dict_classes_db = {
    'diffusion_coefficients': {'einstein_diffusion_coefficients': {'Singular': {}, 'Distinct': {}},
                               'Green_Kubo_Diffusion': {'Singular': {}, 'Distinct': {}}},
    'ionic_conductivity': {},
    'thermal_conductivity': {},
    'coordination_numbers': {'Coordination_Numbers': {}},
    'potential_of_mean_force_values': {'Potential_of_Mean_Force': {}},
    'radial_distribution_function': {},
    'kirkwood_buff_integral': {},
    'structure_factor': {},
    'viscosity': {}
}

"""
Module to contain the structured dict for analysis.

Summary
-------
"""

from mdsuite.analysis.einstein_diffusion_coefficients import EinsteinDiffusionCoefficients
from mdsuite.analysis.green_kubo_diffusion_coefficients import GreenKuboDiffusionCoefficients
from mdsuite.analysis.green_kubo_ionic_conductivity import GreenKuboIonicConductivity
from mdsuite.analysis.einstein_helfand_ionic_conductivity import EinsteinHelfandIonicConductivity
from mdsuite.analysis.radial_distribution_function import RadialDistributionFunction
from mdsuite.analysis.coordination_number_calculation import CoordinationNumbers
from mdsuite.analysis.potential_of_mean_force import PotentialOfMeanForce
from mdsuite.analysis.kirkwood_buff_integrals import KirkwoodBuffIntegral
from mdsuite.analysis.green_kubo_thermal_conductivity import GreenKuboThermalConductivity

dict_classes_computations = {
    'EinsteinDiffusionCoefficients': EinsteinDiffusionCoefficients,
    'GreenKuboDiffusionCoefficients': GreenKuboDiffusionCoefficients,
    'GreenKuboIonicConductivity': GreenKuboIonicConductivity,
    'EinsteinHelfandIonicConductivity': EinsteinHelfandIonicConductivity,
    'RadialDistributionFunction': RadialDistributionFunction,
    'CoordinationNumbers': CoordinationNumbers,
    'PotentialOfMeanForce': PotentialOfMeanForce,
    'KirkwoodBuffIntegral': KirkwoodBuffIntegral,
    'GreenKuboThermalConductivity': GreenKuboThermalConductivity,
}



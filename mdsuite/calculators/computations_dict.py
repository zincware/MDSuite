"""
Module to contain the structured dict for analysis.

Summary
-------
"""

from mdsuite.calculators.coordination_number_calculation import CoordinationNumbers
from mdsuite.calculators.einstein_diffusion_coefficients import EinsteinDiffusionCoefficients
from mdsuite.calculators.einstein_helfand_ionic_conductivity import EinsteinHelfandIonicConductivity
from mdsuite.calculators.einstein_helfand_thermal_conductivity import EinsteinHelfandThermalConductivity
from mdsuite.calculators.flux_thermal import GreenKuboThermalConductivityFlux
from mdsuite.calculators.flux_viscosity import GreenKuboViscosityFlux
from mdsuite.calculators.green_kubo_diffusion_coefficients import GreenKuboDiffusionCoefficients
from mdsuite.calculators.green_kubo_ionic_conductivity import GreenKuboIonicConductivity
from mdsuite.calculators.green_kubo_thermal_conductivity import GreenKuboThermalConductivity
from mdsuite.calculators.green_kubo_viscosity import GreenKuboViscosity
from mdsuite.calculators.kirkwood_buff_integrals import KirkwoodBuffIntegral
from mdsuite.calculators.potential_of_mean_force import PotentialOfMeanForce
from mdsuite.calculators.radial_distribution_function import RadialDistributionFunction
from mdsuite.calculators.structure_factor import StructureFactor

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
    'GreenKuboThermalConductivityFlux': GreenKuboThermalConductivityFlux,
    'StructureFactor': StructureFactor,
    'EinsteinHelfandThermalConductivity': EinsteinHelfandThermalConductivity,
    'GreenKuboViscosityFlux': GreenKuboViscosityFlux,
    'GreenKuboViscosity': GreenKuboViscosity,
}

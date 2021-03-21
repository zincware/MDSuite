"""
A dictionary of possible transformations in th MDSuite code.
"""

from mdsuite.transformations.unwrap_coordinates import CoordinateUnwrapper
from mdsuite.transformations.scale_coordinates import ScaleCoordinates
from mdsuite.transformations.unwrap_via_indices import UnwrapViaIndices
from mdsuite.transformations.ionic_current import IonicCurrent
from mdsuite.transformations.translational_dipole_moment import TranslationalDipoleMoment
from mdsuite.transformations.integrated_heat_current import IntegratedHeatCurrent
from mdsuite.transformations.thermal_flux import ThermalFlux

transformations_dict = {
    'UnwrapCoordinates': CoordinateUnwrapper,
    'ScaleCoordinates': ScaleCoordinates,
    'UnwrapViaIndices': UnwrapViaIndices,
    'IonicCurrent': IonicCurrent,
    'TranslationalDipoleMoment': TranslationalDipoleMoment,
    'IntegratedHeatCurrent': IntegratedHeatCurrent,
    'ThermalFlux': ThermalFlux
}
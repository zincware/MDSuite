"""
A dictionary of possible transformations in th MDSuite code.
"""

from mdsuite.transformations.unwrap_coordinates import CoordinateUnwrapper
from mdsuite.transformations.scale_coordinates import ScaleCoordinates
from mdsuite.transformations.unwrap_via_indices import UnwrapViaIndices
from mdsuite.transformations.ionic_current import IonicCurrent

transformations_dict = {
    'UnwrapCoordinates': CoordinateUnwrapper,
    'ScaleCoordinates': ScaleCoordinates,
    'UnwrapViaIndices': UnwrapViaIndices,
    'IonicCurrent': IonicCurrent
}
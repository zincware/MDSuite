"""
A dictionary of possible transformations in th MDSuite code.
"""

from mdsuite.transformations.unwrap_coordinates import CoordinateUnwrapper
from mdsuite.transformations.scale_coordinates import ScaleCoordinates
from mdsuite.transformations.unwrap_via_indices import UnwrapViaIndices

transformations_dict = {
    'UnwrapCoordinates': CoordinateUnwrapper,
    'ScaleCoordinates': ScaleCoordinates,
    'UnwrapViaIndices': UnwrapViaIndices
}
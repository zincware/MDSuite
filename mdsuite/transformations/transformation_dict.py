"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

A dictionary of possible transformations in th MDSuite code.
"""

from mdsuite.transformations.scale_coordinates import ScaleCoordinates
from mdsuite.transformations.unwrap_via_indices import UnwrapViaIndices
from mdsuite.transformations.ionic_current import IonicCurrent
from mdsuite.transformations.translational_dipole_moment import (
    TranslationalDipoleMoment,
)
from mdsuite.transformations.integrated_heat_current import IntegratedHeatCurrent
from mdsuite.transformations.thermal_flux import ThermalFlux
from mdsuite.transformations.momentum_flux import MomentumFlux
from mdsuite.transformations.kinaci_integrated_heat_current import (
    KinaciIntegratedHeatCurrent,
)
from mdsuite.transformations.wrap_coordinates import CoordinateWrapper
from mdsuite.transformations.unwrap_coordinates import CoordinateUnwrapper

transformations_dict = {
    "UnwrapCoordinates": CoordinateUnwrapper,
    "WrapCoordinates": CoordinateWrapper,
    "ScaleCoordinates": ScaleCoordinates,
    "UnwrapViaIndices": UnwrapViaIndices,
    "IonicCurrent": IonicCurrent,
    "TranslationalDipoleMoment": TranslationalDipoleMoment,
    "IntegratedHeatCurrent": IntegratedHeatCurrent,
    "ThermalFlux": ThermalFlux,
    "MomentumFlux": MomentumFlux,
    "KinaciIntegratedHeatCurrent": KinaciIntegratedHeatCurrent,
}

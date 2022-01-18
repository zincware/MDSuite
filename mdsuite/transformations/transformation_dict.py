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

# TODO remvoe this!
# map_molecules is the only transformation still working with it

from mdsuite.transformations.integrated_heat_current import IntegratedHeatCurrent
from mdsuite.transformations.ionic_current import IonicCurrent
from mdsuite.transformations.kinaci_integrated_heat_current import (
    KinaciIntegratedHeatCurrent,
)
from mdsuite.transformations.momentum_flux import MomentumFlux
from mdsuite.transformations.scale_coordinates import ScaleCoordinates
from mdsuite.transformations.thermal_flux import ThermalFlux
from mdsuite.transformations.translational_dipole_moment import (
    TranslationalDipoleMoment,
)
from mdsuite.transformations.unwrap_coordinates import CoordinateUnwrapper
from mdsuite.transformations.unwrap_via_indices import UnwrapViaIndices
from mdsuite.transformations.wrap_coordinates import CoordinateWrapper

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

# abbreviations are just there to make flake and black happy
from mdsuite.database.mdsuite_properties import mdsuite_properties as mdp
from mdsuite.transformations import (
    integrated_heat_current,
    ionic_current,
    kinaci_integrated_heat_current,
    momentum_flux,
    scale_coordinates,
    thermal_flux,
)
from mdsuite.transformations import translational_dipole_moment as tdp
from mdsuite.transformations import (
    unwrap_coordinates,
    unwrap_via_indices,
    velocity_from_positions,
    wrap_coordinates,
)

"""
Use this dictionary to determine which property can be obtained by which transformation.
Needed for transformation dependency resolution.
"""

property_to_transformation_dict = {
    mdp.integrated_heat_current: integrated_heat_current.IntegratedHeatCurrent,
    mdp.ionic_current: ionic_current.IonicCurrent,
    mdp.kinaci_heat_current: kinaci_integrated_heat_current.KinaciIntegratedHeatCurrent,
    mdp.momentum_flux: momentum_flux.MomentumFlux,
    mdp.positions: [
        scale_coordinates.ScaleCoordinates,
        wrap_coordinates.CoordinateWrapper,
    ],
    mdp.thermal_flux: thermal_flux.ThermalFlux,
    mdp.translational_dipole_moment: tdp.TranslationalDipoleMoment,
    mdp.unwrapped_positions: [
        unwrap_via_indices.UnwrapViaIndices,
        unwrap_coordinates.CoordinateUnwrapper,
    ],
    mdp.velocities_from_positions: velocity_from_positions.VelocityFromPositions,
}

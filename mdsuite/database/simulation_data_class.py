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
from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationProperties:
    """
    A dataclass for the simulation properties.

    Simulation properties refer to the observables measured during a simulation
    for which groups will exist in the HDF5 database.
    """

    temperature = ("Temperature", (None, 1))
    time = ("Time", (None, 1))
    thermal_flux = ("Thermal_Flux", (None, 3))
    stress_viscosity = ("Stress_visc", (None, 3))
    momentum_flux = ("Momentum_Flux", (None, 3))
    ionic_current = ("Ionic_Current", (None, 3))
    translational_dipole_moment = ("Translational_Dipole_Moment", (None, 3))
    positions = ("Positions", (None, None, 3))
    scaled_positions = ("Scaled_Positions", (None, None, 3))
    unwrapped_positions = ("Unwrapped_Positions", (None, None, 3))
    scaled_unwrapped_positions = ("Scaled_Unwrapped_Positions", (None, None, 3))
    velocities = ("Velocities", (None, None, 3))
    forces = ("Forces", (None, None, 3))
    box_images = ("Box_Images", (None, None, 3))
    dipole_orientation_magnitude = ("Dipole_Orientation_Magnitude", (None, None, 1))
    angular_velocity_spherical = ("Angular_Velocity_Spherical", (None, None, 3))
    angular_velocity_non_spherical = ("Angular_Velocity_Non_Spherical", (None, None, 3))
    torque = ("Torque", (None, None, 3))
    integrated_heat_current = ("Integrated_Heat_Current", (None, 3))
    kinaci_heat_current = ("Kinaci_Heat_Current", (None, 3))
    charge = ("Charge", (None, None, 1))
    kinetic_energy = ("KE", (None, None, 1))
    potential_energy = ("PE", (None, None, 1))
    stress = ("Stress", (None, None, 6))

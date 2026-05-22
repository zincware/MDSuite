"""
MDSuite: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Summary
-------
Synthetic stochastic-process helpers used by the Green-Kubo and
Einstein-Helfand calculator tests.

These let the tests construct flux / dipole / moment time series whose
auto-correlation and mean-square-displacement are known *analytically*,
so the calculators can be validated against exact answers rather than
against frozen snapshots.

The module name starts with an underscore so pytest does not try to
collect it as a test module.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np


def ornstein_uhlenbeck_3d(
    n_step: int, dt: float, sigma: float, tau: float, seed: int
) -> np.ndarray:
    """Generate a three-component Ornstein-Uhlenbeck process.

    Each component evolves independently as

        x[i+1] = alpha * x[i] + sigma * sqrt(1 - alpha**2) * eta

    with ``alpha = exp(-dt / tau)`` and ``eta ~ N(0, 1)``. In equilibrium
    the per-component variance is exactly ``sigma**2`` and the ACF is

        <x_i(0) x_i(t)> = sigma**2 * exp(-|t|/tau)

    Parameters
    ----------
    n_step : int
        Number of time steps.
    dt : float
        Time step (in experiment units).
    sigma : float
        Equilibrium standard deviation per Cartesian component.
    tau : float
        Relaxation time (in experiment units).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    flux : np.ndarray, shape (n_step, 3)
        Three-component OU time series.
    """
    rng = np.random.default_rng(seed)
    alpha = np.exp(-dt / tau)
    noise_scale = sigma * np.sqrt(1.0 - alpha**2)

    flux = np.zeros((n_step, 3), dtype=np.float64)
    flux[0] = rng.normal(0.0, sigma, size=3)
    for i in range(1, n_step):
        flux[i] = alpha * flux[i - 1] + noise_scale * rng.normal(size=3)
    return flux


def make_experiment_with_species(
    tmp_path,
    species_names: Iterable[str],
    n_particles_per_species: int,
    box_l: float,
    temperature: float,
    units,
    timestep: float = 1.0,
):
    """Set up a minimal mdsuite project + experiment with registered species.

    Adds two configurations of zero positions so that the species metadata
    (name, n_particles) is fully populated — enough for calculators that
    consume an RDF (and therefore need experiment.species[...].n_particles
    and experiment.volume) but do not themselves re-read positions.

    Returns
    -------
    (project, experiment)
    """
    import mdsuite as mds
    from mdsuite.database.mdsuite_properties import mdsuite_properties
    from mdsuite.database.simulation_database import (
        SpeciesInfo,
        TrajectoryChunkData,
        TrajectoryMetadata,
    )
    from mdsuite.file_io.script_input import ScriptInput

    species_names = list(species_names)
    os.chdir(tmp_path)
    project = mds.Project()
    exp = project.add_experiment(
        "synthetic",
        timestep=timestep,
        temperature=temperature,
        units=units,
    )

    pos_prop = mdsuite_properties.positions
    species_list = [
        SpeciesInfo(name=name, n_particles=n_particles_per_species, properties=[pos_prop])
        for name in species_names
    ]
    metadata = TrajectoryMetadata(
        species_list=species_list,
        n_configurations=2,
        sample_rate=1,
        box_l=[box_l, box_l, box_l],
    )
    data = TrajectoryChunkData(species_list=species_list, chunk_size=2)
    for sp in species_list:
        data.add_data(
            np.zeros((2, n_particles_per_species, 3)),
            0,
            sp.name,
            pos_prop.name,
        )
    exp.add_data(
        ScriptInput(data=data, metadata=metadata, name="synthetic_experiment")
    )
    return project, exp


class SyntheticRDF:
    """Duck-typed stand-in for an mdsuite ``Computation`` carrying an RDF.

    Calculators that consume an RDF (Kirkwood-Buff, potential of mean
    force, coordination numbers, structure factor) accept any object with
    ``data_dict`` and ``computation_parameter`` attributes. This helper
    lets the tests inject a known g(r) without running the RDF calculator
    against synthetic positions.
    """

    def __init__(
        self,
        radii: np.ndarray,
        g_r: np.ndarray,
        species_pair: str,
        number_of_configurations: int = 1,
    ):
        self.data_dict = {
            species_pair: {"x": np.asarray(radii).tolist(), "y": np.asarray(g_r).tolist()}
        }
        self.computation_parameter = {
            "number_of_bins": len(radii),
            "cutoff": float(radii[-1]),
            "number_of_configurations": number_of_configurations,
        }


def gaussian_peak_rdf(
    cutoff: float,
    n_bins: int,
    r_excl: float,
    r_peak: float,
    peak_height: float,
    peak_width: float,
) -> tuple:
    """Build a synthetic g(r) with a smooth excluded region and one Gaussian peak.

    The excluded region is implemented as a logistic ramp centred at
    ``r_excl`` (steepness ``0.05 * r_excl``) rather than a hard step,
    which would create a Savitzky-Golay-filtered spike that confuses
    peak-detection in downstream calculators. The bulk shape is

        bulk(r) = 1 + peak_height * exp(-(r-r_peak)**2 / (2 peak_width**2))

    Returns the (radii, g_r) arrays.
    """
    radii = np.linspace(0.0, cutoff, n_bins)
    ramp = 1.0 / (1.0 + np.exp(-(radii - r_excl) / (0.05 * max(r_excl, 1e-6))))
    bulk = 1.0 + peak_height * np.exp(
        -((radii - r_peak) ** 2) / (2.0 * peak_width**2)
    )
    g_r = ramp * bulk
    return radii, g_r


def cumulative_ou_3d(
    n_step: int, dt: float, sigma: float, tau: float, seed: int
) -> np.ndarray:
    """Generate the cumulative integral of an OU process.

    Used to build a synthetic translational dipole moment or integrated
    heat current for Einstein-Helfand tests. The increments are an OU
    process; the cumulative sum has a mean-square displacement that grows
    diffusively at long times with effective diffusion constant

        D_M = sigma**2 * tau         (per Cartesian component)

    so the 3-component MSD obeys

        <|M(t) - M(0)|**2> -> 6 * D_M * t   as t >> tau.

    Parameters
    ----------
    n_step, dt, sigma, tau, seed
        Same as :func:`ornstein_uhlenbeck_3d`.

    Returns
    -------
    M : np.ndarray, shape (n_step, 3)
        Cumulative OU process. ``M[0] = 0``.
    """
    increments = ornstein_uhlenbeck_3d(n_step, dt, sigma, tau, seed)
    M = np.cumsum(increments * dt, axis=0)
    M -= M[0]  # anchor at zero
    return M

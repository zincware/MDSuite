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
Synthetic-data validation of the Einstein distinct diffusion coefficient
calculator.

Construction: two species ("A" and "B") perform independent random walks.
For independent random walks, the cross-species distinct displacement
correlation <(r_i^A(t) - r_i^A(0))(r_j^B(t) - r_j^B(0))> is zero on
average, so the distinct diffusion coefficient ``D_{AB} = 0``. Within
finite sampling the recovered value should be small compared to the
intrinsic per-particle diffusion ``D = sigma_step**2 / (2 dt)``.

Replaces the prior DataHub-based test which only checked the calculator
ran (no assertions).
"""
import os

import numpy as np
import pytest
import tensorflow as tf

import mdsuite as mds
import mdsuite.utils.units
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.database.simulation_database import (
    SpeciesInfo,
    TrajectoryChunkData,
    TrajectoryMetadata,
)
from mdsuite.file_io.script_input import ScriptInput


def _independent_random_walks(
    n_step: int, n_part: int, dt: float, diff_coeff: float, seed: int
) -> np.ndarray:
    """Independent random walks with per-component diffusion ``diff_coeff``."""
    rng = np.random.default_rng(seed)
    step_std = np.sqrt(2.0 * diff_coeff * dt)
    increments = rng.normal(0.0, step_std, size=(n_step, n_part, 3))
    pos = np.cumsum(increments, axis=0)
    pos[0] = 0.0
    return pos


@pytest.mark.parametrize("desired_memory", (None,))
def test_independent_species(tmp_path, desired_memory):
    """Independent random walks -> distinct diffusion coefficient ~= 0."""
    with mds.utils.helpers.change_memory_fraction(desired_memory=desired_memory):
        time_step = 0.1
        diff_coeff = 1.0
        n_part = 50
        n_step = 4000
        data_range = 200

        os.chdir(tmp_path)
        project = mds.Project()
        units = mdsuite.units.SI
        exp = project.add_experiment(
            "indep_walks",
            timestep=time_step,
            temperature=300.0,
            units=units,
        )

        pos_prop = mdsuite_properties.unwrapped_positions

        species = [
            SpeciesInfo(name="A", n_particles=n_part, properties=[pos_prop]),
            SpeciesInfo(name="B", n_particles=n_part, properties=[pos_prop]),
        ]
        metadata = TrajectoryMetadata(
            species_list=species,
            n_configurations=n_step,
            sample_rate=1,
        )
        data = TrajectoryChunkData(species_list=species, chunk_size=n_step)
        # Two species, two independent seeds.
        data.add_data(
            _independent_random_walks(n_step, n_part, time_step, diff_coeff, seed=1001),
            0,
            "A",
            pos_prop.name,
        )
        data.add_data(
            _independent_random_walks(n_step, n_part, time_step, diff_coeff, seed=2002),
            0,
            "B",
            pos_prop.name,
        )
        exp.add_data(ScriptInput(data=data, metadata=metadata, name="indep_synth"))

        result = exp.run.EinsteinDistinctDiffusionCoefficients(
            plot=False,
            data_range=data_range,
            correlation_time=1,
            species=["A", "B"],
        )

        # The cross-species ("A_B") distinct diffusion coefficient should be
        # small compared to the self-diffusion ``D = diff_coeff`` in SI
        # (length=time=1). The recovered value carries the SI prefactor
        # ``length**2 = 1``, so direct comparison is fine.
        d_ab = result.data_dict["A_B"]["diffusion_coefficient"]
        assert abs(d_ab) < 0.5 * diff_coeff, (
            f"Cross-species distinct diffusion coefficient {d_ab:.3e} is not "
            f"small compared to self-diffusion {diff_coeff:.3e} for independent "
            f"random walks"
        )

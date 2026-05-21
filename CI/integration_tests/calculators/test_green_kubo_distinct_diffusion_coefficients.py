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
Synthetic-data validation of the Green-Kubo distinct diffusion
coefficient calculator.

Two species with statistically independent random velocities have

    <v_i^A(0) . v_j^B(t)> = 0     (i != j, A != B)

so the cross-species GK integral vanishes, and the distinct diffusion
coefficient ``D_{AB} = 0``. Within finite sampling the recovered value
should be small compared to the self-diffusion ``D_AA`` of one species.

Replaces the prior DataHub-based test (which only ran the calculator
without asserting any value).
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


def _independent_random_velocities(
    n_step: int, n_part: int, sigma: float, seed: int
) -> np.ndarray:
    """White-noise velocities (no temporal correlation) for ``n_part`` particles."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=(n_step, n_part, 3))


@pytest.mark.parametrize("desired_memory", (None,))
def test_independent_species(tmp_path, desired_memory):
    """Independent velocities -> GK distinct diffusion coefficient ~= 0."""
    with mds.utils.helpers.change_memory_fraction(desired_memory=desired_memory):
        time_step = 0.1
        sigma = 1.0
        n_part = 50
        n_step = 4000
        data_range = 200

        os.chdir(tmp_path)
        project = mds.Project()
        units = mdsuite.units.SI
        exp = project.add_experiment(
            "indep_vels",
            timestep=time_step,
            temperature=300.0,
            units=units,
        )

        vel_prop = mdsuite_properties.velocities
        species = [
            SpeciesInfo(name="A", n_particles=n_part, properties=[vel_prop]),
            SpeciesInfo(name="B", n_particles=n_part, properties=[vel_prop]),
        ]
        metadata = TrajectoryMetadata(
            species_list=species,
            n_configurations=n_step,
            sample_rate=1,
        )
        data = TrajectoryChunkData(species_list=species, chunk_size=n_step)
        data.add_data(
            _independent_random_velocities(n_step, n_part, sigma, seed=3003),
            0,
            "A",
            vel_prop.name,
        )
        data.add_data(
            _independent_random_velocities(n_step, n_part, sigma, seed=4004),
            0,
            "B",
            vel_prop.name,
        )
        exp.add_data(ScriptInput(data=data, metadata=metadata, name="gk_indep_synth"))

        result = exp.run.GreenKuboDistinctDiffusionCoefficients(
            plot=False,
            data_range=data_range,
            correlation_time=1,
            species=["A", "B"],
        )

        d_aa = abs(result.data_dict["A_A"]["diffusion_coefficient"])
        d_ab = abs(result.data_dict["A_B"]["diffusion_coefficient"])

        # Cross-species correlations should be small relative to the
        # self-correlation of one species.
        assert d_ab < 0.5 * d_aa, (
            f"Cross-species D_AB = {d_ab:.3e} is not small compared to "
            f"D_AA = {d_aa:.3e} for independent random velocities"
        )

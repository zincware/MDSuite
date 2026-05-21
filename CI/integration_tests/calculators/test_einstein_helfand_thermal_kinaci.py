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
Synthetic-data validation of the Einstein-Helfand thermal conductivity
calculator using Kinaci's integrated heat current. Structurally identical
to ``test_einstein_helfand_thermal_conductivity.py`` — the calculator
uses the same MSD-based formula and prefactor, just keyed on a different
input property (``mdsuite_properties.kinaci_heat_current``).

Replaces ``_test_einstein_helfand_thermal_kinaci.py`` (disabled).
"""
import os

import numpy as np
import pytest

import mdsuite as mds
import mdsuite.utils.units
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.database.simulation_database import (
    SpeciesInfo,
    TrajectoryChunkData,
    TrajectoryMetadata,
)
from mdsuite.file_io.script_input import ScriptInput
from mdsuite.utils import DatasetKeys


def _wiener_3d(n_step: int, dt: float, diffusion: float, seed: int) -> np.ndarray:
    """Pure 3-D Wiener process (per-component diffusion ``diffusion``)."""
    rng = np.random.default_rng(seed)
    step_std = np.sqrt(2.0 * diffusion * dt)
    increments = rng.normal(loc=0.0, scale=step_std, size=(n_step, 3))
    M = np.cumsum(increments, axis=0)
    M[0] = 0.0
    return M


@pytest.mark.parametrize("desired_memory", (None, 0.001))
def test_calculator(tmp_path, desired_memory):
    """End-to-end correctness of the Einstein-Helfand thermal Kinaci."""
    with mds.utils.helpers.change_memory_fraction(desired_memory=desired_memory):
        time_step = 0.1
        temperature = 300.0
        box_l = [5.0, 5.0, 5.0]
        volume = float(np.prod(box_l))
        D_Q = volume * mdsuite.utils.units.boltzmann_constant * temperature

        n_step = 20000
        msd_range = 200

        Q = _wiener_3d(n_step=n_step, dt=time_step, diffusion=D_Q, seed=20260526)

        os.chdir(tmp_path)
        project = mds.Project()
        units = mdsuite.units.SI
        exp = project.add_experiment(
            "test_eh_kinaci",
            timestep=time_step,
            temperature=temperature,
            units=units,
        )

        moment_prop = mdsuite_properties.kinaci_heat_current
        observables = SpeciesInfo(
            name=DatasetKeys.OBSERVABLES,
            n_particles=1,
            properties=[moment_prop],
        )
        metadata = TrajectoryMetadata(
            species_list=[observables],
            n_configurations=n_step,
            sample_rate=1,
            box_l=box_l,
        )
        data = TrajectoryChunkData(species_list=[observables], chunk_size=n_step)
        data.add_data(Q.reshape(n_step, 1, 3), 0, observables.name, moment_prop.name)
        exp.add_data(ScriptInput(data=data, metadata=metadata, name="kinaci_synth"))

        result = exp.run.EinsteinHelfandThermalKinaci(
            plot=False,
            data_range=msd_range,
            correlation_time=1,
        )
        system = result.data_dict["System"]

        time_arr = np.array(system["time"])
        msd_arr = np.array(system["msd"])

        prefactor = 1.0 / (volume * temperature * units.boltzmann)
        msd_should_be = prefactor * 6.0 * D_Q * time_arr

        np.testing.assert_allclose(
            msd_arr,
            msd_should_be,
            atol=0.3 * msd_should_be[-1],
            err_msg="E-H Kinaci MSD does not match 6*D_Q*t",
        )

        kappa_recovered = system["thermal_conductivity"]
        kappa_analytic = D_Q / (volume * units.boltzmann * temperature)

        np.testing.assert_allclose(
            kappa_recovered, kappa_analytic, rtol=0.3,
            err_msg=(
                f"Recovered kappa {kappa_recovered:.3e} differs from analytic "
                f"{kappa_analytic:.3e} by more than 30%"
            ),
        )

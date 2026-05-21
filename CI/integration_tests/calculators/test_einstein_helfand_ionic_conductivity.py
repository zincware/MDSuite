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
Synthetic-data validation of the Einstein-Helfand ionic conductivity
calculator.

Construction: the translational dipole moment ``M(t)`` is fed as a
three-dimensional Wiener process with prescribed diffusion constant
``D_M`` per Cartesian component. The mean-square displacement obeys

    <|M(t) - M(0)|**2> = 6 * D_M * t   (sum over 3 components)

so by the Einstein-Helfand formula

    sigma_ionic = (1 / (6 V kB T)) lim_t d/dt <|M(t) - M(0)|**2>
                = D_M / (V kB T)

and the calculator's SI prefactor adds an explicit ``e**2`` factor
(it interprets ``M`` as a moment in length units), giving

    sigma_ionic_calc = e**2 * D_M / (V kB T).

Replaces the prior DataHub-based test, which ran the calculator but
ASSERTED NOTHING ("Test uncertainty is very high!" comment).
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
from mdsuite.utils.units import boltzmann_constant, elementary_charge


def _wiener_3d(n_step: int, dt: float, diffusion: float, seed: int) -> np.ndarray:
    """Pure 3-D Wiener process with per-component diffusion ``diffusion``.

    Each component is the cumulative sum of Gaussian increments with
    variance ``2 * diffusion * dt``. The discrete-time MSD is
    ``<(M[t] - M[0])**2> = 2 * diffusion * t`` per component.
    """
    rng = np.random.default_rng(seed)
    step_std = np.sqrt(2.0 * diffusion * dt)
    increments = rng.normal(loc=0.0, scale=step_std, size=(n_step, 3))
    M = np.cumsum(increments, axis=0)
    M[0] = 0.0
    return M


@pytest.mark.parametrize("desired_memory", (None, 0.001))
def test_calculator(tmp_path, desired_memory):
    """End-to-end correctness of the Einstein-Helfand ionic conductivity."""
    with mds.utils.helpers.change_memory_fraction(desired_memory=desired_memory):
        time_step = 0.1
        D_M = 1.0          # per-component diffusion of the dipole moment
        temperature = 300.0
        box_l = [5.0, 5.0, 5.0]

        n_step = 20000
        msd_range = 200

        M = _wiener_3d(n_step=n_step, dt=time_step, diffusion=D_M, seed=20260524)

        os.chdir(tmp_path)
        project = mds.Project()
        units = mdsuite.units.SI
        exp = project.add_experiment(
            "test_eh_sigma",
            timestep=time_step,
            temperature=temperature,
            units=units,
        )

        moment_prop = mdsuite_properties.translational_dipole_moment
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
        data.add_data(M.reshape(n_step, 1, 3), 0, observables.name, moment_prop.name)
        exp.add_data(ScriptInput(data=data, metadata=metadata, name="eh_sigma_synth"))

        result = exp.run.EinsteinHelfandIonicConductivity(
            plot=False,
            data_range=msd_range,
            correlation_time=1,
        )
        system = result.data_dict["System"]

        # MSD verification: with the calculator's SI prefactor
        # ``e**2 / (V T kB)``, msd[t] should equal prefactor * 6 D_M t.
        time_arr = np.array(system["time"])
        msd_arr = np.array(system["msd"])

        volume = float(np.prod(box_l))
        prefactor = (
            elementary_charge**2 / (volume * temperature * boltzmann_constant)
        )
        msd_should_be = prefactor * 6.0 * D_M * time_arr

        # Loose absolute tolerance scaled to the MSD value at the fit
        # endpoint. Wiener-process MSD fluctuates by ~sqrt(t/tau_eff) and
        # tau_eff -> dt for true Brownian, so 30% near the right end is
        # the right budget.
        np.testing.assert_allclose(
            msd_arr,
            msd_should_be,
            atol=0.3 * msd_should_be[-1],
            err_msg="Einstein-Helfand MSD does not match 6*D_M*t",
        )

        # Aggregated conductivity — E-H calculators store this as a scalar
        # (not a single-element list like the GK ones do).
        sigma_ionic_recovered = system["ionic_conductivity"]
        sigma_ionic_analytic = (
            elementary_charge**2 * D_M / (volume * boltzmann_constant * temperature)
        )

        np.testing.assert_allclose(
            sigma_ionic_recovered, sigma_ionic_analytic, rtol=0.3,
            err_msg=(
                f"Recovered sigma_ionic {sigma_ionic_recovered:.3e} differs "
                f"from analytic {sigma_ionic_analytic:.3e} by more than 30%"
            ),
        )

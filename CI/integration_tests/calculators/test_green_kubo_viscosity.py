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
Synthetic-data validation of the Green-Kubo viscosity calculator
(momentum-flux form).

For an OU momentum flux with relaxation time ``tau`` and per-component
equilibrium variance ``sigma**2``, the analytic Green-Kubo viscosity is

    eta = V / (3 V kB T) * integral_0^inf <P(0) . P(t)> dt
        = sigma**2 * tau / (kB * T)

(the volume cancels; ``P`` here is the momentum flux per volume, not the
microscopic stress).

Replaces ``_test_green_kubo_viscosity.py``, which was disabled because it
relied on a downloaded NaCl trajectory and a snapshot-comparison.
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

from ._synthetic_signals import ornstein_uhlenbeck_3d


@pytest.mark.parametrize("desired_memory", (None, 0.001))
def test_calculator(tmp_path, desired_memory):
    """End-to-end correctness of the GK viscosity calculator."""
    with mds.utils.helpers.change_memory_fraction(desired_memory=desired_memory):
        time_step = 0.1
        sigma = 1.0
        tau = 10.0
        temperature = 300.0
        box_l = [5.0, 5.0, 5.0]

        n_step = 20000
        data_range = 200

        flux = ornstein_uhlenbeck_3d(
            n_step=n_step, dt=time_step, sigma=sigma, tau=tau, seed=20260521
        )

        os.chdir(tmp_path)
        project = mds.Project()
        units = mdsuite.units.SI
        exp = project.add_experiment(
            "test_eta",
            timestep=time_step,
            temperature=temperature,
            units=units,
        )

        flux_prop = mdsuite_properties.momentum_flux
        observables = SpeciesInfo(
            name=DatasetKeys.OBSERVABLES,
            n_particles=1,
            properties=[flux_prop],
        )
        metadata = TrajectoryMetadata(
            species_list=[observables],
            n_configurations=n_step,
            sample_rate=1,
            box_l=box_l,
        )
        data = TrajectoryChunkData(species_list=[observables], chunk_size=n_step)
        data.add_data(flux.reshape(n_step, 1, 3), 0, observables.name, flux_prop.name)
        exp.add_data(ScriptInput(data=data, metadata=metadata, name="eta_synth"))

        result = exp.run.GreenKuboViscosity(
            plot=False,
            data_range=data_range,
            correlation_time=1,
            integration_range=data_range,
        )
        system = result.data_dict["System"]

        acf_summed = np.array(system["acf"])
        time_arr = np.array(system["time"])

        n_windows = n_step - data_range
        acf_estimate = acf_summed / (n_windows * data_range)
        acf_should_be = 3 * sigma**2 * np.exp(-time_arr / tau)

        # t=0 amplitude is the sum of per-component variances. Same
        # OU-process sampling-noise budget as the thermal conductivity test.
        np.testing.assert_allclose(
            acf_estimate[0], acf_should_be[0], rtol=0.2,
            err_msg="ACF at t=0 (= sum of variances) is off",
        )

        decay_window = int(tau / time_step)
        np.testing.assert_allclose(
            acf_estimate[:decay_window],
            acf_should_be[:decay_window],
            atol=0.3 * acf_should_be[0],
            err_msg="ACF early-time decay does not match exp(-t/tau)",
        )

        # The calculator's prefactor in SI is
        #   1 / (3 (N-1) T kB V),
        # the per-window integrated ACF for an OU flux is
        #   integral_0^T  3 sigma^2 exp(-t/tau) dt -> 3 sigma^2 tau
        # so kappa -> sigma^2 tau / (T kB V).
        volume = float(np.prod(box_l))
        eta_recovered = system["viscosity"][0]
        eta_analytic = sigma**2 * tau / (temperature * units.boltzmann * volume)

        np.testing.assert_allclose(
            eta_recovered, eta_analytic, rtol=0.3,
            err_msg=(
                f"Recovered viscosity {eta_recovered:.3e} differs from analytic "
                f"{eta_analytic:.3e} by more than 30%"
            ),
        )

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
Synthetic-data validation of the Green-Kubo ionic conductivity calculator.

Construction: feed an Ornstein-Uhlenbeck ionic-current time series with
known per-component variance ``sigma**2`` and relaxation time ``tau``.
The analytic GK ionic conductivity is

    sigma_ionic = (1 / (3 V kB T)) * integral_0^inf <J(0) . J(t)> dt
                = sigma**2 * tau / (V kB T)

(the elementary-charge factor cancels in SI because the calculator uses
``elementary_charge**2`` inside its prefactor and we feed the current
directly in SI units).

Replaces the previous DataHub-snapshot test.
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

from ._synthetic_signals import ornstein_uhlenbeck_3d


@pytest.mark.parametrize("desired_memory", (None, 0.001))
def test_calculator(tmp_path, desired_memory):
    """End-to-end correctness of the GK ionic conductivity calculator."""
    with mds.utils.helpers.change_memory_fraction(desired_memory=desired_memory):
        time_step = 0.1
        sigma = 1.0
        tau = 10.0
        temperature = 300.0
        box_l = [5.0, 5.0, 5.0]

        n_step = 20000
        data_range = 200

        current = ornstein_uhlenbeck_3d(
            n_step=n_step, dt=time_step, sigma=sigma, tau=tau, seed=20260523
        )

        os.chdir(tmp_path)
        project = mds.Project()
        units = mdsuite.units.SI
        exp = project.add_experiment(
            "test_sigma_ionic",
            timestep=time_step,
            temperature=temperature,
            units=units,
        )

        flux_prop = mdsuite_properties.ionic_current
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
        data.add_data(current.reshape(n_step, 1, 3), 0, observables.name, flux_prop.name)
        exp.add_data(ScriptInput(data=data, metadata=metadata, name="sigma_synth"))

        result = exp.run.GreenKuboIonicConductivity(
            plot=False,
            data_range=data_range,
            correlation_time=1,
            # Default integration_range = data_range - 1; explicit values
            # equal to data_range overrun ``cumulative_trapezoid`` by one.
        )
        system = result.data_dict["System"]

        acf_summed = np.array(system["acf"])
        time_arr = np.array(system["time"])

        # The ionic conductivity calculator already averages internally
        # (``self.acf_array /= self.count``). With ``correlation_time=1`` the
        # ensemble loop slides by one step, yielding ``n_step - data_range``
        # windows. The TFP biased estimator returns ``(1/T) sum x x`` and the
        # calculator does NOT multiply by data_range (unlike GK thermal /
        # viscosity), so the stored ACF is already
        # ``count * <J(0).J(t)> / count = <J(0).J(t)>`` — no further
        # normalisation needed.
        acf_estimate = acf_summed
        acf_should_be = 3 * sigma**2 * np.exp(-time_arr / tau)

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

        # The GK ionic conductivity calculator's prefactor in SI is
        #   e**2 * length**2 / (3 kB T V time)
        # and it multiplies the cumulative integral at index
        # ``integration_range - 1``. For an OU current with
        # ``<J(0).J(t)> = 3 sigma**2 exp(-t/tau)``, the integral to t >> tau
        # is ``3 sigma**2 tau``, so
        #   sigma_ionic -> e**2 * sigma**2 * tau / (kB T V)
        # in SI.
        volume = float(np.prod(box_l))
        sigma_ionic_recovered = system["ionic_conductivity"][0]
        sigma_ionic_analytic = (
            elementary_charge**2 * sigma**2 * tau
            / (boltzmann_constant * temperature * volume)
        )

        np.testing.assert_allclose(
            sigma_ionic_recovered, sigma_ionic_analytic, rtol=0.3,
            err_msg=(
                f"Recovered sigma_ionic {sigma_ionic_recovered:.3e} differs "
                f"from analytic {sigma_ionic_analytic:.3e} by more than 30%"
            ),
        )

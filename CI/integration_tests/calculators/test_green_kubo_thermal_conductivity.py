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
Synthetic-data validation of the Green-Kubo thermal conductivity calculator.

Approach: generate a heat-flux time series whose autocorrelation function is
known analytically — a three-component Ornstein-Uhlenbeck process. Feed it
into the calculator and verify that the recovered autocorrelation matches
the exact ACF.

For an OU process with relaxation time ``tau`` and equilibrium variance
``sigma**2`` (per Cartesian component, independent across components):

    <J_i(0) J_i(t)> = sigma**2 * exp(-|t|/tau)

So the system ACF (summed over the three components):

    <J(0) . J(t)>   = 3 * sigma**2 * exp(-|t|/tau)

and the Green-Kubo thermal conductivity is

    kappa = (1 / (3 V kB T**2)) * integral_0^inf <J(0) . J(t)> dt
          = sigma**2 * tau / (V * kB * T**2)

Replaces the prior snapshot-comparison test
(``_test_green_kubo_thermal_conductivity.py``), which is disabled.
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
    """End-to-end correctness of the GK thermal conductivity calculator.

    Generates an OU heat flux with known relaxation time and variance,
    pushes it into a synthetic experiment, runs the calculator, and asserts
    that the recovered autocorrelation matches the analytic OU ACF.
    """
    with mds.utils.helpers.change_memory_fraction(desired_memory=desired_memory):
        # --- Physical parameters ------------------------------------------------
        time_step = 0.1
        sigma = 1.0      # std-dev of heat-flux components (in experiment units)
        tau = 10.0       # OU relaxation time (in experiment units)
        temperature = 300.0
        box_l = [5.0, 5.0, 5.0]  # box side length; volume = 125

        n_step = 20000   # well above the data_range so we get many windows
        data_range = 200

        # --- Synthetic trajectory ----------------------------------------------
        flux = ornstein_uhlenbeck_3d(
            n_step=n_step, dt=time_step, sigma=sigma, tau=tau, seed=20260520
        )

        os.chdir(tmp_path)
        project = mds.Project()
        # Use SI to keep the unit-conversion factors trivial (length = time =
        # energy = 1). boltzmann is small (~1.4e-23) so we work with a very
        # small numerical kappa, which is fine for the relative-error checks.
        units = mdsuite.units.SI
        exp = project.add_experiment(
            "test_kappa",
            timestep=time_step,
            temperature=temperature,
            units=units,
        )

        flux_prop = mdsuite_properties.thermal_flux
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
        # Data must be shape (n_step, n_particles=1, n_dims=3).
        data = TrajectoryChunkData(species_list=[observables], chunk_size=n_step)
        data.add_data(flux.reshape(n_step, 1, 3), 0, observables.name, flux_prop.name)
        proc = ScriptInput(data=data, metadata=metadata, name="kappa_synth")
        exp.add_data(proc)

        result = exp.run.GreenKuboThermalConductivity(
            plot=False,
            data_range=data_range,
            correlation_time=1,
            integration_range=data_range,
        )
        # System-property calculators key their output under "System".
        system = result.data_dict["System"]

        # --- Verify the autocorrelation shape ---------------------------------
        # The calculator stores ``self.jacf`` as the SUMMED autocorrelation
        # (TFP biased estimator multiplied by data_range, summed over the
        # three Cartesian components, summed over all windows).
        acf_summed = np.array(system["acf"])
        time_arr = np.array(system["time"])

        # With ``correlation_time=1`` the ensemble loop uses a sliding window
        # of width ``data_range``, yielding ``n_step - data_range`` windows.
        # ``tfp.stats.auto_correlation`` with ``normalize=False`` returns
        # ``(1/T) sum_i x[i] x[i+k]`` (biased estimator), and the calculator
        # multiplies that by ``data_range``. So
        # ``acf_summed / (n_windows * data_range) == <J(0) . J(t)>``.
        n_windows = n_step - data_range
        acf_estimate = acf_summed / (n_windows * data_range)

        # Analytic three-component OU ACF: 3 * sigma^2 * exp(-t/tau).
        acf_should_be = 3 * sigma**2 * np.exp(-time_arr / tau)

        # At t=0 the empirical and analytic ACFs should match within
        # sampling noise. For an OU process the standard error on the
        # variance scales like ``sqrt(tau / n_step)`` per component, so
        # with n_step=20000 and tau=10*dt=1 (=> tau/n_step=5e-4) plus the
        # OU correlation lengthening the effective sample count, ~20% is
        # the right budget.
        np.testing.assert_allclose(
            acf_estimate[0], acf_should_be[0], rtol=0.2,
            err_msg="ACF at t=0 (= sum of variances) is off",
        )

        # Over the first relaxation time the ACF should decay exponentially.
        decay_window = int(tau / time_step)
        np.testing.assert_allclose(
            acf_estimate[:decay_window],
            acf_should_be[:decay_window],
            atol=0.3 * acf_should_be[0],
            err_msg="ACF early-time decay does not match exp(-t/tau)",
        )

        # --- Verify the aggregated kappa --------------------------------------
        # ``thermal_conductivity`` is the mean across windows of
        # ``prefactor * trapz(window_jacf, time)``; ``uncertainty`` is the SEM.
        kappa_recovered = system["thermal_conductivity"][0]
        volume = float(np.prod(box_l))
        kappa_analytic = (
            sigma**2 * tau / (volume * units.boltzmann * temperature**2)
        )

        # 30% tolerance: stochastic integral of an OU ACF over a finite
        # window picks up tail noise, especially with `integration_range`
        # close to several relaxation times.
        np.testing.assert_allclose(
            kappa_recovered, kappa_analytic, rtol=0.3,
            err_msg=(
                f"Recovered kappa {kappa_recovered:.3e} differs from analytic "
                f"{kappa_analytic:.3e} by more than 30%"
            ),
        )

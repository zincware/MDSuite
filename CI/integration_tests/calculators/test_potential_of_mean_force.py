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
Analytical validation of the Potential of Mean Force calculator.

The PMF is defined pointwise as

    w(r) = -k_B T * ln(g(r))

(modulo a unit-conversion factor the calculator applies). With a
synthetic g(r) that has a single Gaussian peak above unity, the PMF
should be negative near the peak (attractive) and vanish where
``g(r) = 1``.
"""
import dataclasses

import numpy as np
import pytest

import mdsuite as mds
import mdsuite.utils.units
from mdsuite.utils.units import boltzmann_constant

from ._synthetic_signals import (
    SyntheticRDF,
    gaussian_peak_rdf,
    make_experiment_with_species,
)


@pytest.fixture
def synthetic_experiment(tmp_path):
    units = dataclasses.replace(mdsuite.units.SI, length=1e-9)
    _, exp = make_experiment_with_species(
        tmp_path=tmp_path,
        species_names=["A"],
        n_particles_per_species=100,
        box_l=10.0,
        temperature=300.0,
        units=units,
    )
    return exp


def test_gaussian_peak_pmf(synthetic_experiment):
    """A single-peak g(r) should produce a PMF dip at the peak location."""
    cutoff = 5.0
    n_bins = 400
    r_peak = 1.5
    peak_height = 2.0           # g(r_peak) = 3
    peak_width = 0.2

    radii, g_r = gaussian_peak_rdf(
        cutoff=cutoff,
        n_bins=n_bins,
        r_excl=0.7,
        r_peak=r_peak,
        peak_height=peak_height,
        peak_width=peak_width,
    )
    # The PMF peak-finder requires at least two peaks so it can define a
    # first minimum between them. Add a small second peak.
    r_peak2 = 3.0
    g_r = g_r + 0.5 * np.exp(-((radii - r_peak2) ** 2) / (2.0 * (peak_width * 1.5) ** 2))
    rdf = SyntheticRDF(radii, g_r, species_pair="A_A")

    result = synthetic_experiment.run.PotentialOfMeanForce(
        rdf_data=rdf,
        plot=False,
        savgol_order=2,
        savgol_window_length=11,
        number_of_shells=1,
    )

    pomf = np.array(result.data_dict["A_A"]["pomf"])
    r_axis = np.array(result.data_dict["A_A"]["r"])

    peak_idx = int(np.argmin(np.abs(r_axis - r_peak)))
    pmf_at_peak = pomf[peak_idx]

    # Analytic: w(r_peak) = -kT * ln(1 + peak_height) * J->eV conversion
    kT_in_J = boltzmann_constant * 300.0
    j_to_ev = 6.242e8
    pmf_analytic = -kT_in_J * np.log(1.0 + peak_height) * j_to_ev

    np.testing.assert_allclose(
        pmf_at_peak,
        pmf_analytic,
        rtol=0.15,
        err_msg=(
            f"PMF at the peak ({pmf_at_peak:.3e}) does not match the analytic "
            f"{pmf_analytic:.3e} = -kT ln(1+A) within 15%"
        ),
    )
    assert pmf_at_peak < 0, (
        f"PMF at the peak should be negative; got {pmf_at_peak:.3e}"
    )

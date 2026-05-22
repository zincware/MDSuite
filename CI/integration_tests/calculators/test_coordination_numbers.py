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
Analytical validation of the Coordination Numbers calculator.

The first-shell coordination number is defined as

    N_1 = rho * 4 pi * integral_0^{r_min} r**2 g(r) dr

where ``r_min`` is the first minimum of g(r) after the first peak.

We synthesise a g(r) with a single Gaussian peak above unity and verify
that the calculator's reported ``CN_1`` matches the closed-form integral
to within the peak-detection and Savgol-smoothing tolerances.
"""
import dataclasses

import numpy as np
import pytest
from scipy.integrate import quad

import mdsuite as mds
import mdsuite.utils.units

from ._synthetic_signals import (
    SyntheticRDF,
    gaussian_peak_rdf,
    make_experiment_with_species,
)


@pytest.fixture
def synthetic_setup(tmp_path):
    units = dataclasses.replace(mdsuite.units.SI, length=1e-9)
    box_l = 10.0
    n_particles = 100
    _, exp = make_experiment_with_species(
        tmp_path=tmp_path,
        species_names=["A"],
        n_particles_per_species=n_particles,
        box_l=box_l,
        temperature=300.0,
        units=units,
    )
    return exp, n_particles, box_l


def test_gaussian_peak_coordination(synthetic_setup):
    """First-shell CN of a single-Gaussian-peak g(r) matches the integral.

    Uses a g(r) shaped like (excluded volume + one Gaussian peak +
    asymptote to 1). The first peak is at ``r_peak``; the first minimum
    sits between ``r_peak`` and the next "peak" of the constant tail.
    With no second peak in the synthetic g(r), the calculator's
    peak-finding logic needs at least one more peak to define the first
    minimum, which we add as a slight bump at ``r_peak2``.
    """
    exp, n_particles, box_l = synthetic_setup

    cutoff = 5.0
    n_bins = 600
    r_excl = 0.7
    r_peak = 1.5
    peak_height = 3.0
    peak_width = 0.2

    radii, g_r = gaussian_peak_rdf(
        cutoff=cutoff,
        n_bins=n_bins,
        r_excl=r_excl,
        r_peak=r_peak,
        peak_height=peak_height,
        peak_width=peak_width,
    )
    # Add a second, smaller Gaussian peak so the calculator's
    # find_peaks logic can identify a first minimum between them.
    r_peak2 = 3.0
    g_r += 0.5 * np.exp(-((radii - r_peak2) ** 2) / (2.0 * (peak_width * 1.5) ** 2))
    g_r = np.where(radii < r_excl, 0.0, g_r)

    rdf = SyntheticRDF(radii, g_r, species_pair="A_A")
    result = exp.run.CoordinationNumbers(
        rdf_data=rdf,
        plot=False,
        savgol_order=2,
        savgol_window_length=11,
        number_of_shells=1,
    )

    cn_recovered = result.data_dict["A_A"]["CN_1"]

    # Analytic value: integrate 4 pi r**2 g(r) from r_excl to r_min,
    # multiply by density. Estimate r_min by the location of the
    # minimum of g(r) between the two peaks.
    interior = (radii > r_peak) & (radii < r_peak2)
    r_min = float(radii[interior][int(np.argmin(g_r[interior]))])

    def integrand(r):
        return 4.0 * np.pi * r**2 * np.interp(r, radii, g_r)

    # Volume in nm**3 (length=1 nm under our chosen units).
    volume_nm3 = box_l**3
    density = n_particles / volume_nm3
    cn_analytic, _ = quad(integrand, r_excl, r_min, limit=500)
    cn_analytic *= density

    np.testing.assert_allclose(
        cn_recovered, cn_analytic, rtol=0.25,
        err_msg=(
            f"CN_1 {cn_recovered:.3f} differs from analytic {cn_analytic:.3f}"
            f" by more than 25%"
        ),
    )

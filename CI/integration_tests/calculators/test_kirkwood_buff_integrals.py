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
Analytical validation of the Kirkwood-Buff Integral calculator.

The KBI is defined as

    G_AB(r) = 4 pi * integral_0^r (g_AB(r') - 1) r'**2 dr'

so feeding in an *analytically known* g(r) gives an analytically known
G_AB(r). Two limiting cases are checked:

* Ideal gas (``g(r) = 1`` everywhere) -> ``G(r) = 0`` for all r.
* Hard-shell-only RDF (``g(r) = 0`` for ``r < r_excl``, ``g(r) = 1``
  elsewhere) -> ``G(r) = -4 pi r_excl**3 / 3`` for ``r >> r_excl``,
  which is just minus the excluded volume.
"""
import dataclasses

import numpy as np
import pytest

import mdsuite as mds
import mdsuite.utils.units

from ._synthetic_signals import SyntheticRDF, make_experiment_with_species


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


def test_ideal_gas_kbi(synthetic_experiment):
    """g(r) = 1 everywhere -> KBI = 0 everywhere."""
    radii = np.linspace(0.0, 5.0, 200)
    g_r = np.ones_like(radii)
    rdf = SyntheticRDF(radii, g_r, species_pair="A_A")

    result = synthetic_experiment.run.KirkwoodBuffIntegral(
        rdf_data=rdf,
        plot=False,
        savgol_order=2,
        savgol_window_length=11,
    )

    kbi = np.array(result.data_dict["A_A"]["kb_integral"])
    np.testing.assert_allclose(
        kbi,
        np.zeros_like(kbi),
        atol=1e-6,
        err_msg="Ideal-gas KBI is not zero across the whole r range",
    )


def test_hard_shell_kbi(synthetic_experiment):
    """g(r) = 0 below r_excl, =1 above -> KBI(r) -> -4 pi r_excl**3 / 3."""
    r_excl = 1.0
    radii = np.linspace(0.0, 5.0, 500)
    g_r = np.where(radii < r_excl, 0.0, 1.0)
    rdf = SyntheticRDF(radii, g_r, species_pair="A_A")

    result = synthetic_experiment.run.KirkwoodBuffIntegral(
        rdf_data=rdf,
        plot=False,
        savgol_order=2,
        savgol_window_length=11,
    )

    kbi = np.array(result.data_dict["A_A"]["kb_integral"])
    # The calculator integrates from radii[1:] and returns
    # ``cumulative_trapezoid`` (length len(radii)-3 in our case).
    r_axis = radii[3 : 3 + len(kbi)]
    kbi_analytic_far = -4.0 / 3.0 * np.pi * r_excl**3

    far_idx = r_axis > 1.5 * r_excl
    np.testing.assert_allclose(
        kbi[far_idx],
        kbi_analytic_far * np.ones(far_idx.sum()),
        atol=0.2 * abs(kbi_analytic_far),
        err_msg=(
            "Hard-shell KBI does not asymptote to -4 pi r_excl**3 / 3"
        ),
    )

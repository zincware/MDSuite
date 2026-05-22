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
Analytical validation of the Structure Factor calculator.

For a single-species ideal gas (``g(r) = 1`` everywhere), the partial
structure factor reduces to the constant ``1 + 4 pi rho * integral_0^inf
(g(r) - 1) r**2 sin(qr)/(qr) dr / 2 = 0.5`` in this calculator's
convention. Combined with the same-species weighting factor (``factor=2``
in :meth:`_compute_total_structure_factor`) and the trivial single-species
form-factor weight of 1, the total structure factor is exactly ``1.0``
for all ``q``.

Replaces the double-disabled ``__test_structure_factor.py`` (the previous
test contained a hard-coded Windows path).
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
    # The structure-factor calculator looks up atomic form factors by
    # element name; use a real symbol that exists in form_fac_coeffs.csv.
    _, exp = make_experiment_with_species(
        tmp_path=tmp_path,
        species_names=["Na"],
        n_particles_per_species=100,
        box_l=10.0,
        temperature=300.0,
        units=units,
    )
    return exp


def test_ideal_gas_structure_factor(synthetic_experiment):
    """Ideal-gas single-species g(r) = 1 -> total S(q) = 1.0 for all q."""
    radii = np.linspace(0.0, 5.0, 400)
    g_r = np.ones_like(radii)
    rdf = SyntheticRDF(radii, g_r, species_pair="Na_Na")

    result = synthetic_experiment.run.StructureFactor(
        rdf_data=rdf,
        plot=False,
        resolution=300,
    )

    total = np.array(result.data_dict["System"]["S"])

    np.testing.assert_allclose(
        total,
        np.ones_like(total),
        atol=1e-6,
        err_msg="Ideal-gas total S(q) is not identically 1.0",
    )

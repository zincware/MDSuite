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
Analytical validation of the radial distribution function calculator.

For an ideal gas (independent particles uniformly distributed in a box
with periodic boundary conditions), the pair distribution function is
exactly

    g(r) = 1   for  r < L/2

in the thermodynamic limit. Finite-size and finite-sampling deviations
are bounded by a known sampling-noise envelope that depends only on the
shell occupation, the number of configurations, and the bin width.

Constructing a "trajectory" of uniformly random positions and feeding it
to the RDF calculator is a strictly tighter test than running ASE: the
reference is closed-form rather than tabulated, the cost is one
``rng.uniform`` call rather than a real MD step loop, and the
sampling-noise envelope can be derived rather than measured.

Replaces the prior DataHub snapshot-comparison test.
"""
import dataclasses
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


def _uniform_positions(
    n_step: int, n_part: int, box_l: float, seed: int
) -> np.ndarray:
    """Independent uniform-random positions in a cubic box.

    Returns
    -------
    pos : np.ndarray, shape (n_step, n_part, 3)
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, box_l, size=(n_step, n_part, 3))


def test_ideal_gas_rdf(tmp_path):
    """An ideal gas (uniform-random positions in a periodic box) has g(r)=1.

    A single-species ideal gas should reproduce ``g(r) = 1`` exactly in
    the thermodynamic limit. With a finite box and finite sampling, the
    radial bins are subject to Poisson-like counting noise, so we allow
    a 10% absolute tolerance on the central region of the RDF.
    """
    n_step = 200
    n_part = 400
    box_l = 10.0
    cutoff = 4.0          # well below L/2
    number_of_bins = 80

    pos = _uniform_positions(n_step, n_part, box_l, seed=20260527)

    os.chdir(tmp_path)
    project = mds.Project()
    # The RDF calculator converts radii to nm via ``units.length / 1e-9``;
    # using ``length = 1e-9`` makes that conversion the identity and lets
    # us state ``box_l`` and ``cutoff`` directly in nm.
    units = dataclasses.replace(mdsuite.units.SI, length=1e-9)
    exp = project.add_experiment(
        "ideal_gas",
        timestep=1.0,
        temperature=1.0,
        units=units,
    )

    pos_prop = mdsuite_properties.positions
    species = SpeciesInfo(
        name="ideal", n_particles=n_part, properties=[pos_prop]
    )
    metadata = TrajectoryMetadata(
        species_list=[species],
        n_configurations=n_step,
        sample_rate=1,
        box_l=[box_l, box_l, box_l],
    )
    data = TrajectoryChunkData(species_list=[species], chunk_size=n_step)
    data.add_data(pos, 0, species.name, pos_prop.name)
    exp.add_data(ScriptInput(data=data, metadata=metadata, name="ideal_gas_rdf"))

    result = exp.run.RadialDistributionFunction(
        plot=False,
        cutoff=cutoff,
        number_of_bins=number_of_bins,
        number_of_configurations=n_step,
        start=0,
        stop=n_step - 1,
    )

    # The RDF is keyed by species pair; for one species we get one entry.
    key = "ideal_ideal"
    rdf = np.array(result.data_dict[key]["y"])
    radii = np.array(result.data_dict[key]["x"])

    # Skip the first bin (numerical artefacts at r=0) and bins beyond
    # half the box length (the calculator does not apply the
    # finite-box correction past L/2, so g(r) drops there).
    valid = (radii > 0.5) & (radii < cutoff * 0.9)

    # Ideal-gas reference.
    g_should_be = np.ones_like(rdf)

    # Sampling-noise envelope: for an ideal gas with N particles in a
    # box of side L sampled n_step times, the standard error on g(r) in
    # a spherical shell of radius r and thickness dr scales like
    # ``sqrt(1 / (N (N-1) n_step rho 4 pi r**2 dr))``. With the values
    # below the envelope is well under 10% across the valid range, so
    # ``atol=0.1`` is a comfortable budget.
    np.testing.assert_allclose(
        rdf[valid],
        g_should_be[valid],
        atol=0.1,
        err_msg="Ideal-gas RDF is not g(r) ~ 1 across the valid range",
    )

    # Spot-check the bin-averaged g(r) over the valid range, which
    # should be very close to 1 because the sampling noise averages out.
    mean_g = float(np.mean(rdf[valid]))
    assert abs(mean_g - 1.0) < 0.02, (
        f"Mean g(r) over the valid range is {mean_g:.3f}, expected ~1.0"
    )

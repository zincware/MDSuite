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
Analytical validation of the Angular Distribution Function calculator.

For independent particles in a periodic box (ideal gas), the angle
``theta`` subtended at one particle by two of its neighbours within a
cutoff is uniformly distributed in cos(theta). When binning by ``theta``
itself, the expected ADF follows the Jacobian

    p(theta) ~ sin(theta)

with the peak at 90 degrees and zeros at 0 and 180. The calculator's
ADF normalisation also includes a distance-weighting ``norm_power``,
but for sufficiently small cutoff (so the distance dependence is mild)
and ``norm_power=0`` the shape is dominated by the geometric Jacobian.

Replaces the prior DataHub snapshot test.
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


def test_ideal_gas_adf(tmp_path):
    """Ideal-gas ADF should peak near 90 degrees with the sin(theta) shape."""
    n_step = 20
    n_part = 200
    box_l = 5.0  # nm-equivalent under the units below
    cutoff = 2.0

    rng = np.random.default_rng(20260528)
    pos = rng.uniform(0.0, box_l, size=(n_step, n_part, 3))

    os.chdir(tmp_path)
    project = mds.Project()
    # length = 1 nm keeps the calculator's nm-conversion as identity.
    units = dataclasses.replace(mdsuite.units.SI, length=1e-9)
    exp = project.add_experiment(
        "ideal_gas_adf",
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
    exp.add_data(ScriptInput(data=data, metadata=metadata, name="ideal_gas_adf"))

    result = exp.run.AngularDistributionFunction(
        plot=False,
        cutoff=cutoff,
        number_of_configurations=n_step,
        start=0,
        stop=n_step - 1,
        number_of_bins=60,
        norm_power=0,        # no distance weighting; shape dominated by Jacobian
        use_tf_function=False,
    )

    key = "ideal_ideal_ideal"
    angles_deg = np.array(result.data_dict[key]["angle"])
    adf = np.array(result.data_dict[key]["adf"])

    # The peak of the histogram should be near 90 degrees.
    peak_deg = angles_deg[np.argmax(adf)]
    assert 70.0 < peak_deg < 110.0, (
        f"ADF peak is at {peak_deg:.1f} degrees, expected near 90"
    )

    # The histogram should be roughly symmetric about 90 degrees: tail
    # masses on the two sides should be comparable within sampling noise.
    left = adf[angles_deg < 90].sum()
    right = adf[angles_deg >= 90].sum()
    asymmetry = abs(left - right) / (left + right)
    assert asymmetry < 0.2, (
        f"ADF is too asymmetric about 90 deg (left/right balance = {asymmetry:.3f})"
    )

    # Endpoints (near 0 and 180 degrees) should be small compared to the
    # peak — the sin(theta) Jacobian vanishes there.
    end_max = max(adf[0:3].max(), adf[-3:].max())
    assert end_max < 0.5 * adf.max(), (
        "ADF does not approach zero at theta -> 0 or 180 (sin(theta) Jacobian violated)"
    )

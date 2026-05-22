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
Exercises the standalone-calculator API: a calculator is constructed once
(without any experiment), then applied to one or more experiments via
``experiment.run(calc)`` / ``project.run(calc)``.

These tests pin the new contract that comes out of the calculator refactor:

* ``Calculator.__init__`` takes no ``experiment`` / ``experiments`` argument.
* The same calculator instance can be reused across experiments without state
  bleed-through.
* The legacy ``experiment.run.<Calculator>(**kwargs)`` API still works through
  the back-compat shim and produces identical results to the new API.
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
from mdsuite.utils.testing import assertDeepAlmostEqual


def _make_uniform_gas_experiment(tmp_path):
    """Spin up a minimal experiment with uniform-random positions.

    Reused by all the tests in this module to keep them independent of
    external trajectory downloads.
    """
    n_step = 80
    n_part = 200
    box_l = 10.0

    rng = np.random.default_rng(20260530)
    pos = rng.uniform(0.0, box_l, size=(n_step, n_part, 3))

    os.chdir(tmp_path)
    project = mds.Project()
    units = dataclasses.replace(mdsuite.units.SI, length=1e-9)
    project.add_experiment(
        "ideal", timestep=1.0, temperature=300.0, units=units
    )
    exp = project.experiments["ideal"]

    pos_prop = mdsuite_properties.positions
    species = SpeciesInfo(name="ideal", n_particles=n_part, properties=[pos_prop])
    metadata = TrajectoryMetadata(
        species_list=[species],
        n_configurations=n_step,
        sample_rate=1,
        box_l=[box_l, box_l, box_l],
    )
    data = TrajectoryChunkData(species_list=[species], chunk_size=n_step)
    data.add_data(pos, 0, species.name, pos_prop.name)
    exp.add_data(ScriptInput(data=data, metadata=metadata, name="ideal_gas"))
    return project, exp


def test_calculator_constructable_without_experiment():
    """A calculator can be built without any experiment or project around."""
    calc = mds.GreenKuboDiffusionCoefficients(data_range=500, plot=False)

    assert calc.experiment is None
    assert calc._user_data_range == 500
    assert calc.plot is False


def test_experiment_run_accepts_calculator_instance(tmp_path):
    """``experiment.run(calc)`` returns a single ``db.Computation``."""
    _, exp = _make_uniform_gas_experiment(tmp_path)

    calc = mds.RadialDistributionFunction(plot=False, number_of_configurations=50)
    result = exp.run(calc)

    assert result is not None
    assert result.data_dict, "RDF must produce a non-empty data dict"


def test_project_run_accepts_calculator_instance(tmp_path):
    """``project.run(calc)`` returns a ``{experiment_name: result}`` dict."""
    project, _ = _make_uniform_gas_experiment(tmp_path)

    calc = mds.RadialDistributionFunction(plot=False, number_of_configurations=50)
    results = project.run(calc)

    assert isinstance(results, dict)
    assert "ideal" in results


def test_calculator_clears_experiment_between_runs(tmp_path):
    """The calculator must release its experiment reference after ``run`` returns."""
    _, exp = _make_uniform_gas_experiment(tmp_path)

    calc = mds.RadialDistributionFunction(plot=False, number_of_configurations=50)
    exp.run(calc)

    assert calc.experiment is None, (
        "Calculator must release its experiment reference after run() returns"
    )


def test_new_api_and_legacy_shim_produce_equal_results(tmp_path):
    """``experiment.run.X(**kwargs)`` and ``experiment.run(calc)`` agree."""
    _, exp = _make_uniform_gas_experiment(tmp_path)

    legacy = exp.run.RadialDistributionFunction(
        plot=False, number_of_configurations=50
    )

    calc = mds.RadialDistributionFunction(plot=False, number_of_configurations=50)
    fresh = exp.run(calc)

    assertDeepAlmostEqual(legacy.data_dict, fresh.data_dict, decimal=6)


def _add_uniform_species(project, name, rng, n_step=60, n_part=150, box_l=10.0):
    """Attach a uniform-random ideal-gas trajectory to an existing project."""
    project.add_experiment(
        name,
        timestep=1.0,
        temperature=300.0,
        units=dataclasses.replace(mdsuite.units.SI, length=1e-9),
    )
    exp = project.experiments[name]

    pos = rng.uniform(0.0, box_l, size=(n_step, n_part, 3))
    pos_prop = mdsuite_properties.positions
    species = SpeciesInfo(name="ideal", n_particles=n_part, properties=[pos_prop])
    metadata = TrajectoryMetadata(
        species_list=[species],
        n_configurations=n_step,
        sample_rate=1,
        box_l=[box_l, box_l, box_l],
    )
    data = TrajectoryChunkData(species_list=[species], chunk_size=n_step)
    data.add_data(pos, 0, species.name, pos_prop.name)
    exp.add_data(ScriptInput(data=data, metadata=metadata, name=f"{name}_data"))


def test_project_run_parallel_matches_serial(tmp_path):
    """``project.run(calc, parallel=True)`` matches the serial result.

    Two independent projects (separate tmp dirs so the SQLite caches do
    not cross-talk), each with two experiments. One project runs the
    calculator serially; the other runs it through the thread-pool
    parallel path. The recovered RDFs must agree.
    """
    rng_a = np.random.default_rng(20260601)
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial_dir.mkdir()
    parallel_dir.mkdir()

    os.chdir(serial_dir)
    serial_project = mds.Project()
    _add_uniform_species(serial_project, "alpha", np.random.default_rng(1))
    _add_uniform_species(serial_project, "beta", np.random.default_rng(2))
    serial_calc = mds.RadialDistributionFunction(plot=False, number_of_configurations=40)
    serial_results = serial_project.run(serial_calc, parallel=False)

    os.chdir(parallel_dir)
    parallel_project = mds.Project()
    _add_uniform_species(parallel_project, "alpha", np.random.default_rng(1))
    _add_uniform_species(parallel_project, "beta", np.random.default_rng(2))
    parallel_calc = mds.RadialDistributionFunction(plot=False, number_of_configurations=40)
    parallel_results = parallel_project.run(parallel_calc, parallel=True)

    assert set(serial_results) == set(parallel_results)
    for name in serial_results:
        assertDeepAlmostEqual(
            serial_results[name].data_dict,
            parallel_results[name].data_dict,
            decimal=6,
        )

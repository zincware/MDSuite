"""
MDSuite: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
Module to test scaling coefficients.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest
from zinchub import DataHub

import mdsuite
import mdsuite.transformations


def _build_atomwise(data_scaling: int, system: bool = False):
    """
    Build a numpy array of atom-wise data in steps of MBs.

    Parameters
    ----------
    data_scaling : int
            Number of atoms in the data e.g. zeroth array of the data. 1 atom is 1/10
            of a MB of data.
    system : bool
            If true, the returned array should be (n_confs, 3)

    Returns
    -------
    data_array : np.ones
            A numpy array of ones that matches close to 1/10 * data_scaling MBs in
            size (~98%).
    Notes
    -----
    TODO: When moved to (confs, n_atoms, dim), this will need to be updated to take the
          first column as atoms otherwise the memory scaling will be wrong.

    """
    if system:
        return np.ones((data_scaling * 4096, 3))
    else:
        return np.ones((data_scaling, 4096, 3))


@pytest.fixture(scope="session")
def traj_file(tmp_path_factory) -> str:
    """Download trajectory file into a temporary directory and keep it for all tests"""
    temporary_path = tmp_path_factory.getbasetemp()

    NaCl = DataHub(url="https://github.com/zincware/DataHub/tree/main/NaCl_gk_i_q")
    NaCl.get_file(path=temporary_path)

    return (temporary_path / NaCl.file_raw).as_posix()


@pytest.fixture()
def mdsuite_project(tmp_path) -> mdsuite.Project:
    """
    Build an MDSuite project with all data stored in a temp directory for easier
    cleanup after the test.

    Returns
    -------
    project : mdsuite.Project
            MDSuite project to be used in the tests.
    """
    project = mdsuite.Project(storage_path=tmp_path.as_posix())
    project.add_experiment("NaCl", simulation_data=traj_file)

    scaling_sizes = [10, 100, 500, 1000]

    return project


def get_memory_usage(database: str, callable_name: str) -> float:
    """
    Get the memory used from the dumped sql database.

    Parameters
    ----------
    database : str
            Path to the sqlite database that will be read.
    callable_name : str
            Name of the function being measured and therefore, what memory value to
            return.

    Returns
    -------
    memory : float
            memory used during the calculation.
    """
    with sqlite3.connect(database) as db:
        data = pd.read_sql_query("SELECT * from TEST_METRICS", db)

    data = data.loc[data["ITEM"] == callable_name]

    return data["MEM_USAGE"]


def test_rdf_memory(mdsuite_project):
    """
    Test the memory of the RDF.

    Parameters
    ----------
    mdsuite_project : mdsuite.Project
            An mdsuite project with stored files in a tmp directory.

    Returns
    -------

    """
    memory_array = np.zeros((2,))
    mdsuite_project.run.RadialDistributionFunction(plot=False)
    memory = get_memory_usage("pymon.db", test_rdf_memory.__name__)
    memory_array[0] = memory

    print(memory_array)

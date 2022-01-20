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
Example script for using the visualizer.
"""
import os
import tempfile

from zinchub import DataHub

import mdsuite as mds


def load_data():
    """
    Load simulation data from the server.

    Returns
    -------
    Will store simulation data locally for the example.
    """
    argon = DataHub(url="https://github.com/zincware/DataHub/tree/main/Ar/Ar_dft")
    argon.get_file(path=".")


def run_example():
    """
    Run the visualizer example.

    Returns
    -------

    """
    project = mds.Project("Argon_Example")
    project.add_experiment(
        name="argon",
        timestep=0.1,
        temperature=85.0,
        units="metal",
        simulation_data="Ar_dft_short.extxyz",
    )
    project.experiments.argon.run_visualization()


if __name__ == "__main__":
    """
    Run the example.
    """
    temp_dir = tempfile.TemporaryDirectory()
    os.chdir(temp_dir.name)
    load_data()  # load the data.
    run_example()  # run the example.
    os.chdir("..")
    temp_dir.cleanup()

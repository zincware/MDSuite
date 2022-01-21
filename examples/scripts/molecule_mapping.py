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
    bmim = DataHub(url="https://github.com/zincware/DataHub/tree/main/Bmim_BF4")
    bmim.get_file(path=".")


def run_example():
    """
    Run the bmim_bf4 example.

    Returns
    -------
    Runs the example.
    """
    project = mds.Project("bmim_bf4_example")
    project.add_experiment(
        name="bmim_bf4",
        timestep=0.1,
        temperature=100.0,
        units="real",
        simulation_data="bmim_bf4.lammpstraj",
    )
    project.experiments.bmim_bf4.run.UnwrapViaIndices()
    project.run.MolecularMap(
        molecules={
            "bmim": {"smiles": "CCCCN1C=C[N+](+C1)C", "amount": 50, "cutoff": 1.9},
            "bf4": {"smiles": "[B-](F)(F)(F)F", "amount": 50, "cutoff": 2.4},
        }
    )
    project.run.AngularDistributionFunction(
        start=0, stop=100, number_of_configurations=10, cutoff=3.0
    )
    project.experiments.bmim_bf4.run_visualization(molecules=False)

    print("Tutorial complete....... Files being deleted now.")


if __name__ == "__main__":
    """
    Collect and run the code.
    """
    temp_dir = tempfile.TemporaryDirectory()
    os.chdir(temp_dir.name)
    load_data()  # load the data.
    run_example()  # run the example.
    os.chdir("..")
    temp_dir.cleanup()

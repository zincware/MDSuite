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
import mdsuite as mds
import urllib.request
import shutil
import gzip
import os
import tempfile
from mdsuite.transformations.map_molecules import MolecularMap


def load_data():
    """
    Load simulation data from the server.

    Returns
    -------
    Will store simulation data locally for the example.
    """
    base_url = "https://github.com/zincware/Datahub/raw/main/bmim_bf4.lammpstrj"
    filename, headers = urllib.request.urlretrieve(
        f"{base_url}.gz", filename="bmim_bf4.lammpstrj.gz"
    )
    with gzip.open(filename, "rb") as f_in:
        with open("bmim_bf4.lammpstraj", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def run_example():
    """
    Run the bmim_bf4 example.

    Returns
    -------
    Nul.
    """
    project = mds.Project("bmim_bf4_example")
    project.add_experiment(
        experiment="bmim_bf4",
        timestep=0.1,
        temperature=100.0,
        units="real",
        data="bmim_bf4.lammpstraj",
    )

    project.experiments.bmim_bf4.perform_transformation("UnwrapCoordinates")
    mapper = MolecularMap(
        project.experiments.bmim_bf4,
        molecules={
            "bmim": {"smiles": "CCCCN1C=C[N+](+C1)C", "cutoff": 1.9, "amount": 50},
            "bf4": {"smiles": "[B-](F)(F)(F)F", "cutoff": 2.4, "amount": 50},
        },
    )
    mapper.run_transformation()

    project.run.SpatialDistributionFunction(
        species=["bmim", "bf4"],
        r_min=1.0,
        r_max=9.0,
        number_of_configurations=50,
        n_bins=1000,
        start=0,
        stop=80,
    )

    #project.experiments.bmim_bf4.run_visualization(molecules=True)

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

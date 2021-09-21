"""
A script to compute the Green-Kubo diffusion coefficients.

Examples
--------
>>> python3 green_kubo_diffusion.py
"""
import mdsuite as mds
import urllib.request
import shutil
import gzip
import os
import tempfile


def load_data():
    """
    Load simulation data from the server.

    Returns
    -------
    Will store simulation data locally for the example.
    """
    base_url = 'https://github.com/zincware/ExampleData/raw/main/NaCl_gk_i_q.lammpstraj'
    filename, headers = urllib.request.urlretrieve(f'{base_url}.gz',
                                                   filename='NaCl_gk_i_q.lammpstraj.gz')
    with gzip.open(filename, 'rb') as f_in:
        with open('NaCl_gk_i_q.lammpstraj', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def run_example():
    """
    Run the GK example.

    Returns
    -------

    """
    project = mds.Project("GK_Diffusion_Example")
    project.add_experiment(
        experiment='NaCl_GK',
        timestep=0.002,
        temperature=1400.0,
        units='metal',
        data='NaCl_gk_i_q.lammpstraj'
    )
    project.run.GreenKuboDiffusionCoefficients(
        data_range=100, plot=True, correlation_time=10
    )
    project.run.EinsteinDiffusionCoefficients(
        data_range=100, correlation_time=10, plot=True
    )
    project.run.EinsteinHelfandIonicConductivity(
        data_range=100, correlation_time=10, plot=True
    )

    project.run.RadialDistributionFunction(
        number_of_configurations=100, start=0, stop=101, plot=True
    )
    project.run.CoordinationNumbers(plot=True)
    project.run.PotentialOfMeanForce(plot=True)


if __name__ == '__main__':
    """
    Run the example.
    """
    temp_dir = tempfile.TemporaryDirectory()
    os.chdir(temp_dir.name)
    load_data()  # load the data.
    run_example()  # run the example.
    os.chdir('..')
    temp_dir.cleanup()

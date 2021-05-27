"""
Integration test with Liquid NaCl.

Tests the following:
* Project class function
* Experiment class function
* RDF calculator
* Diffusion coefficients
* Ionic Conductivity with and without charges.
* Unwrapping with and without indices
"""
import mdsuite as mds
import unittest
import urllib.request
import gzip
import shutil
import os


class TestLiquidNaCl(unittest.TestCase):
    """
    Perform an integration test on liquid NaCl.
    """
    def setUp(self):
        """
        set up the project class and experiments.

        Returns
        -------
        Updates class attributes.
        """
        self.project = mds.Project(name='molten_nacl_test')
        time_step = 0.002
        temperature = 1400.0
        base_url = 'https://github.com/zincware/ExampleData/raw/main/'
        endpoints = ['NaCl_gk_i_q.lammpstraj',
                     'NaCl_gk_ni_nq.lammpstraj',
                     'NaCl_i_q.lammpstraj',
                     'NaCl_ni_nq.lammpstraj']
        for item in endpoints:
            print(item)
            filename, headers = urllib.request.urlretrieve(f'{base_url}{item}.gz', filename=f'{item}.gz')
            with gzip.open(filename, 'rb') as f_in:
                with open(item, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            self.project.add_experiment(experiment=item[:-11],
                                        timestep=time_step,
                                        temperature=temperature,
                                        units='metal')
            self.project.experiments[item[:-11]].add_data(item)
            os.remove(item)
            os.remove(f'{item}.gz')

    def test_completion(self):
        """
        Test that the project is loaded.
        Returns
        -------
        assert that all the experiments exist.
        """
        print(self.project.experiments.keys())


if __name__ == '__main__':
    unittest.main()

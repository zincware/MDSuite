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
import unittest
import urllib.request
import gzip
import os
import numpy as np
import shutil

# TODO run in temporary directory!


class TestLiquidNaCl(unittest.TestCase):
    """
    Perform an integration test on liquid NaCl.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        set up the project class and experiments.

        Returns
        -------
        Updates class attributes.
        """
        cls.project = mds.Project(name='molten_nacl_test')
        time_step = 0.002
        temperature = 1400.0
        base_url = 'https://github.com/zincware/ExampleData/raw/main/'
        cls.endpoints = ['NaCl_gk_i_q.lammpstraj',
                         'NaCl_gk_ni_nq.lammpstraj',
                         'NaCl_i_q.lammpstraj',
                         'NaCl_ni_nq.lammpstraj']
        for item in cls.endpoints:
            filename, headers = urllib.request.urlretrieve(f'{base_url}{item}.gz',
                                                           filename=f'{item}.gz')
            with gzip.open(filename, 'rb') as f_in:
                with open(item, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            cls.project.add_experiment(experiment=item[:-11],
                                       timestep=time_step,
                                       temperature=temperature,
                                       units='metal')
            cls.project.experiments[item[:-11]].add_data(item)
            cls.project.experiments[item[:-11]].set_charge("Na", 1)
            cls.project.experiments[item[:-11]].set_charge("Cl", -1)
            os.remove(item)
            os.remove(f'{item}.gz')

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Remove files generated.
        Returns
        -------
        Removes the project directory.
        """
        shutil.rmtree('molten_nacl_test_MDSuite_Project', ignore_errors=True)

    def test_completion(self):
        """
        Test that the project is loaded.
        Returns
        -------
        assert that all the experiments exist.
        """
        reference = [item[:-11] for item in self.endpoints]
        comparison = list(self.project.experiments.keys())
        assert np.array_equal(reference, comparison)

    def test_einstein_diffusion_coefficients(self):
        """
        Test that einstein diffusion coefficients work as expected. This will
        include assessing how the unwrapping works for both indices and from
        the MDSuite box-hopping implementation.

        Returns
        -------
        Asserts the following:
        * Self-diffusion coefficients match to expected value.
        * Box-indices and box-hopping unwrapping methods yield the same value.
        * All unwrapping, einstein computations, and fitting methods work
          together.
        """
        Na_ref = 1.46e-8
        Cl_ref = 1.41e-8
        reference_experiments = ['NaCl_i_q', 'NaCl_ni_nq']
        for item in reference_experiments:
            self.project.experiments[item].run_computation.EinsteinDiffusionCoefficients(plot=False,
                                                                                         data_range=300,
                                                                                         correlation_time=1,
                                                                                         save=True)
        dat_Na = self.project.get_properties({'analysis': 'Einstein_Self_Diffusion_Coefficients',
                                              'subjects': ["Na"]})
        dat_Cl = self.project.get_properties({'analysis': 'Einstein_Self_Diffusion_Coefficients',
                                              'subjects': ["Cl"]})

        Na_diff = [[dat_Na[item][0].data[0].x] for item in reference_experiments]
        Cl_diff = [[dat_Cl[item][0].data[0].x] for item in reference_experiments]
        np.testing.assert_almost_equal(Na_diff[0][0], Na_ref, 8)
        np.testing.assert_almost_equal(Na_diff[1][0], Na_ref, 8)
        np.testing.assert_almost_equal(Na_diff[0][0], Na_diff[1][0], 8)
        np.testing.assert_almost_equal(Cl_diff[0][0], Cl_ref, 8)
        np.testing.assert_almost_equal(Cl_diff[1][0], Cl_ref, 8)
        np.testing.assert_almost_equal(Cl_diff[0][0], Cl_diff[1][0], 8)

    def test_einstein_ionic_conductivity(self):
        """
        Test that Einstein-Helfand ionic conductivity work as expected.
        This will include assessing how the unwrapping works for both indices
        and from the MDSuite box-hopping implementation. It will also ensure
        that the different approaches for charge inclusion result in the same
        values.

        Returns
        -------
        Asserts the following:
        * Ionic conductivity values match to expected value.
        * Box-indices and box-hopping unwrapping methods yield the same value.
        * All unwrapping, einstein computations, and fitting methods work
          together.
        """
        reference_experiments = ['NaCl_i_q', 'NaCl_ni_nq']
        ic_ref = 365.0
        for item in reference_experiments:
            self.project.experiments[item].run_computation.EinsteinHelfandIonicConductivity(plot=False,
                                                                                            data_range=50,
                                                                                            correlation_time=1,
                                                                                            save=True)
        dat = self.project.get_properties({'analysis': 'Einstein_Helfand_Ionic_Conductivity'})

        ic = [[dat[item][0].data[0].x] for item in reference_experiments]
        np.testing.assert_almost_equal(ic[0][0], ic[1][0], -1)
        np.testing.assert_almost_equal(ic[0][0], ic_ref, -1)
        np.testing.assert_almost_equal(ic[1][0], ic_ref, -1)

    def test_green_kubo_diffusion_coefficients(self):
        """
        Test that green-kubo diffusion coefficients work as expected. This will
        include assessing how the unwrapping works for both indices and from
        the MDSuite box-hopping implementation.

        Returns
        -------
        Asserts the following:
        * Self-diffusion coefficients match to expected value.
        """
        Na_ref = 1.78e-8
        Cl_ref = 1.43e-8
        reference_experiments = ['NaCl_gk_i_q', 'NaCl_gk_ni_nq']
        for item in reference_experiments:
            self.project.experiments[item].run_computation.GreenKuboDiffusionCoefficients(plot=True,
                                                                                          data_range=500,
                                                                                          integration_range=350,
                                                                                          correlation_time=1,
                                                                                          save=True)
        dat_Na = self.project.get_properties({'analysis': 'Green_Kubo_Self_Diffusion_Coefficients',
                                              'subjects': ["Na"]})
        dat_Cl = self.project.get_properties({'analysis': 'Green_Kubo_Self_Diffusion_Coefficients',
                                              'subjects': ["Cl"]})
        Na_diff = [[dat_Na[item][0].data[0].x] for item in reference_experiments]
        Cl_diff = [[dat_Cl[item][0].data[0].x] for item in reference_experiments]
        np.testing.assert_almost_equal(Na_diff[0][0], Na_ref, 5)
        np.testing.assert_almost_equal(Na_diff[1][0], Na_ref, 5)
        np.testing.assert_almost_equal(Na_diff[0][0], Na_diff[1][0], 5)
        np.testing.assert_almost_equal(Cl_diff[0][0], Cl_ref, 5)
        np.testing.assert_almost_equal(Cl_diff[1][0], Cl_ref, 5)
        np.testing.assert_almost_equal(Cl_diff[0][0], Cl_diff[1][0], 5)

    def test_green_kubo_ionic_conductivity(self):
        """
        Test that Green-Kubo ionic conductivity work as expected. This will
        include assessing how the unwrapping works for both indices and from
        the MDSuite box-hopping implementation. It will also ensure that the
        different approaches for charge inclusion result in the same values.

        Returns
        -------
        Asserts the following:
        * Ionic conductivity values match to expected value.
        * All unwrapping, einstein computations, and fitting methods work
          together.
        * Charge computation and matrix computation works the same.
        """
        reference_experiments = ['NaCl_gk_i_q', 'NaCl_gk_ni_nq']
        ic_ref = 2048

        for item in reference_experiments:
            self.project.experiments[item].run_computation.GreenKuboIonicConductivity(plot=True,
                                                                                      data_range=500,
                                                                                      integration_range = 300,
                                                                                      correlation_time=1,
                                                                                      save=True)
        dat = self.project.get_properties({'analysis': 'Green_Kubo_Ionic_Conductivity'})
        ic = [[dat[item][0].data[0].x] for item in reference_experiments]
        np.testing.assert_almost_equal(ic[0][0], ic[1][0], -1)
        np.testing.assert_almost_equal(ic[0][0], ic_ref, -1)
        np.testing.assert_almost_equal(ic[1][0], ic_ref, -1)


if __name__ == '__main__':
    unittest.main()
    shutil.rmtree('molten_nacl_test_MDSuite_Project', ignore_errors=True)

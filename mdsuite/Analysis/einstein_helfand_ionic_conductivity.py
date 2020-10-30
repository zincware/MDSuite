"""
Class for the calculation of the Einstein-Helfand ionic conductivity.

Author: Samuel Tovey

Description: This module contains the code for the Einstein-Helfand ionic conductivity class. This class is called by
the Experiment class and instantiated when the user calls the Experiment.einstein_helfand_ionic_conductivity method.
The methods in class can then be called by the Experiment.einstein_helfand_ionic_conductivity method and all necessary
calculations performed.
"""

# Python standard packages
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Import user packages
from tqdm import tqdm
import torch

# Import MDSuite modules
import mdsuite.constants as constants

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class _EinsteinHelfandIonicConductivity:
    """ Class for the Einstein-Helfand Ionic Conductivity """

    def __init__(self, obj, data_range, plot):
        """ Python constructor """

        self.parent = obj
        self.data_range = data_range
        self.plot = plot
        self.number_of_configurations = self.parent.number_of_configurations - self.parent.number_of_configurations % \
                                        self.parent.batch_size
        self.loop_range = self.number_of_configurations - data_range - 1
        self.correlation_time = 1

    def _autocorrelation_time(self):
        """ Calculate dipole moment autocorrelation time

        When performing this analysis, the sampling should occur over the autocorrelation time of the positions in the
        system. This method will calculate what this time is and sample over it to ensure uncorrelated samples.
        """
        pass

    def _calculate_translational_dipole(self):
        """ Calculate the translational dipole of the system """

        # Load the particle positions and sum
        dipole_moment = torch.from_numpy(self.parent.load_matrix("Unwrapped_Positions")).sum(1)

        # Build the charge tensor for assignment
        system_charges = [self.parent.species[atom]['charge'][0] for atom in self.parent.species]
        charge_tuple = ()
        for charge in system_charges:
            charge_tuple += (torch.ones(self.number_of_configurations, 3))*charge
        charge_tensor = torch.stack(charge_tuple)

        dipole_moment *= charge_tensor  # Multiply the dipole moment tensor by the system charges
        dipole_moment = dipole_moment.sum(0)  # Calculate the final dipole moments

        return dipole_moment

    def _calculate_ionic_conductivity(self):
        """ Calculate the conductivity """

        dipole_moment_tensor = self._calculate_translational_diplole()  # Calculate the dipole moment
        dipole_msd_tensor = torch.zeros(self.data_range, 3)  # Initialize the msd tensor

        # Construct mask tensor









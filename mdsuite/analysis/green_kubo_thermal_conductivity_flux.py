"""
Class for the calculation of the einstein diffusion coefficients.

Author: Francisco Torres ; Samuel Tovey

Description: This module contains the code for the thermal conductivity class. This class is called by the
Experiment class and instantiated when the user calls the ... method.
The methods in class can then be called by the ... method and all necessary
calculations performed.
"""

import warnings

# Python standard packages
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Import user packages
from tqdm import tqdm

# MDSuite packages
import mdsuite.utils.constants as constants

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class _EinsteinDiffusionCoefficients:
    """ Class for the Einstein diffusion coefficient implementation

    additional attrbs:
        plot
        singular
        distinct
        species
        data_range
    """

    def __init__(self, obj, plot=False, data_range=500):
        self.parent = obj
        self.plot = plot
        self.data_range = data_range
        self.time = self.time = np.linspace(0.0, self.data_range * self.parent.time_step * self.parent.sample_rate,
                                            self.data_range)

        raise NotImplementedError  # This code does not work yet

    def _autocorrelation_time(self):
        """ Claculate the flux autocorrelation time to ensure correct sampling """
        raise NotImplementedError

    def _compute_thermal_conductivity(self):
        """ Compute the thermal conductivity """

        if self.plot:
            averaged_jacf = np.zeros(self.data_range)

        # prepare the prefactor for the integral
        numerator = 1
        denominator = 3 * (self.data_range / 2 - 1) * self.parent.temperature ** 2 * constants.boltzmann_constant \
                      * self.parent.volume * self.parent.units['length'] ** 3
        # not sure why I need the /2 in data range...
        prefactor = numerator / denominator

        flux = self.parent.load_matrix()

        loop_range = len(flux) - self.data_range - 1  # Define the loop range
        sigma = []

        # main loop for computation
        for i in tqdm(range(loop_range)):
            jacf = np.zeros(2 * self.data_range - 1)  # Define the empty JACF array
            jacf += (signal.correlate(flux[:, 0][i:i + self.data_range],
                                      flux[:, 0][i:i + self.data_range],
                                      mode='full', method='fft') +
                     signal.correlate(flux[:, 1][i:i + self.data_range],
                                      flux[:, 1][i:i + self.data_range],
                                      mode='full', method='fft') +
                     signal.correlate(flux[:, 2][i:i + self.data_range],
                                      flux[:, 2][i:i + self.data_range],
                                      mode='full', method='fft'))

            # Cut off the second half of the acf
            jacf = jacf[int((len(jacf) / 2)):]
            if self.plot:
                averaged_jacf += jacf

            integral = np.trapz(jacf, x=self.time)
            sigma.append(integral)

        sigma = prefactor * np.array(sigma)

        if self.plot:
            averaged_jacf /= max(averaged_jacf)
            plt.plot(self.time, averaged_jacf)
            plt.xlabel("Time (s)")
            plt.ylabel("Normalized Current Autocorrelation Function")
            plt.savefig(f"GK_Cond_{self.parent.temperature}.pdf", )
            plt.show()

        print(f"Green-Kubo Ionic Conductivity at {self.parent.temperature}K: {np.mean(sigma)} +- "
              f"{np.std(sigma) / np.sqrt(len(sigma))} W/m/K")

        self._save_class()  # Update class state

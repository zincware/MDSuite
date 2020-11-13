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
import warnings

# Import user packages
from tqdm import tqdm
import torch

# Import MDSuite modules

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

        raise NotImplementedError

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
        raise NotImplementedError

"""
 # Fill the dipole moment msd matrix
        for i in tqdm(range(loop_range)):
            for j in range(3):
                dipole_moment_msd[j] += (dipole_moment[i:i + data_range, j] - dipole_moment[i][j]) ** 2

        dipole_msd = np.array(np.array(dipole_moment_msd[0]) +
                              np.array(dipole_moment_msd[1]) +
                              np.array(dipole_moment_msd[2]))  # Calculate the net msd

        # Initialize the time
        time = np.linspace(0.0, data_range * self.sample_rate * self.time_step, len(dipole_msd[0]))

        sigma_array = []  # Initialize and array for the conductivity calculations
        # Loop over different fit ranges to generate an array of conductivities, from which a value can be calculated
        for i in range(100):
            # Create the data range
            start = np.random.randint(int(0.1 * len(dipole_msd[0])), int(0.60 * len(dipole_msd[0])))
            stop = np.random.randint(int(1.4 * start), int(1.65 * start))

            # Calculate the value and append the array
            popt, pcov = curve_fit(meta_functions.linear_fitting_function, time[start:stop], dipole_msd[0][start:stop])
            sigma_array.append(popt[0])

        # Define the multiplicative prefactor of the calculation
        denominator = (6 * self.temperature * (self.volume * self.length_unit ** 3) * constants.boltzmann_constant) * \
                      self.time_unit * loop_range
        numerator = (self.length_unit ** 2) * (constants.elementary_charge ** 2)
        prefactor = numerator / denominator

        sigma = prefactor * np.mean(sigma_array)
        sigma_error = prefactor * (np.sqrt(np.var(sigma_array)) / np.sqrt(len(sigma_array)))

        if plot:
            plt.plot(time, dipole_msd[0])
            plt.xlabel("Time")
            plt.ylabel("Dipole Mean Square Displacement")
            plt.savefig(f"EHCond_{self.temperature}.pdf", format='pdf', dpi=600)
            plt.show()

        print(f"Einstein-Helfand Conductivity at {self.temperature}K: {sigma / 100} +- {sigma_error / 100} S/cm")

        self._save_class()  # Update class state
"""









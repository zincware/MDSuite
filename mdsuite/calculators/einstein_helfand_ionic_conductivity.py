"""
Class for the calculation of the Einstein-Helfand ionic conductivity.

Summary
-------
This class is called by the Experiment class and instantiated when the user calls the
Experiment.einstein_helfand_ionic_conductivity method. The methods in class can then be called by the
Experiment.einstein_helfand_ionic_conductivity method and all necessary calculations performed.
"""

import matplotlib.pyplot as plt
import warnings
from scipy.optimize import curve_fit
import numpy as np
import os

# Import user packages
from tqdm import tqdm
import tensorflow as tf
import h5py as hf

# Import MDSuite modules
import mdsuite.utils.meta_functions as meta_functions
from mdsuite.utils.units import elementary_charge, boltzmann_constant
from mdsuite.calculators.calculator import Calculator

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class EinsteinHelfandIonicConductivity(Calculator):
    """
    Class for the Einstein-Helfand Ionic Conductivity

    Attributes
    ----------
    obj :  object
            Experiment class to call from
    plot : bool
            if true, plot the data
    species : list
            Which species to perform the analysis on
    data_range :
            Number of configurations to use in each ensemble
    save :
            If true, data will be saved after the analysis
    x_label : str
            X label of the data when plotted
    y_label : str
            Y label of the data when plotted
    analysis_name : str
            Name of the analysis
    loaded_property : str
            Property loaded from the database for the analysis
    batch_loop : int
            Number of ensembles in each batch
    time : np.array
            Array of the time.
    correlation_time : int
            Correlation time of the property being studied. This is used to ensure ensemble sampling is only performed
            on uncorrelated samples. If this is true, the error extracted form the calculation will be correct.

    """

    def __init__(self, obj, plot=True, data_range=500, save=True,
                 x_label='Time (s)', y_label='MSD (m^2/s)', analysis_name='einstein_helfand_ionic_conductivity'):
        """
        Python constructor

        Parameters
        ----------
        obj :  object
            Experiment class to call from
        plot : bool
                if true, plot the data
        species : list
                Which species to perform the analysis on
        data_range :
                Number of configurations to use in each ensemble
        save :
                If true, data will be saved after the analysis
        x_label : str
                X label of the data when plotted
        y_label : str
                Y label of the data when plotted
        analysis_name : str
                Name of the analysis
        """

        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name)  # parse to the parent class

        self.loaded_property = 'Unwrapped_Positions'  # Property to be loaded for the analysis
        self.batch_loop = None                        # Number of ensembles in a batch
        self.parallel = True                          # Set the parallel attribute
        self.tensor_choice = True                     # Load data as a tensor

        self.correlation_time = 1                     # Correlation time of the current
        self.species = list(obj.species)              # species on which to perform the analysis

        self.database_group = 'ionic_conductivity'    # Which database group to save the data in

        # Time array for the calculations
        self.time = np.linspace(0.0, self.data_range * self.parent.time_step * self.parent.sample_rate, self.data_range)

        # Check for unwrapped coordinates and unwrap if not stored already.

        with hf.File(os.path.join(obj.database_path, 'database.hdf5'), "r+") as database:
            for item in self.species:
                # Unwrap the positions if they need to be unwrapped
                if "Unwrapped_Positions" not in database[item]:
                    print("Unwrapping coordinates")
                    obj.perform_transformation('UnwrapCoordinates', species=[item])  # Unwrap the coordinates
                    print("Coordinate unwrapping finished, proceeding with analysis")

    def _autocorrelation_time(self):
        """
        Calculate dipole moment autocorrelation time

        When performing this analysis, the sampling should occur over the autocorrelation time of the positions in the
        system. This method will calculate what this time is and sample over it to ensure uncorrelated samples.
        """
        pass

    def _calculate_integrated_current(self, data):
        """
        Calculate the translational dipole of the system

        This method will calculate the translational dipole moment of a single batch of data.

        Parameters
        ----------
        data : list
                A list of tensor corresponding to the positions of the particles.

        Returns
        -------
        dipole_moment : tf.tensor
                Return the dipole moment for the batch
        """

        counter = 0                                             # set a counter variable
        for tensor in data:                                     # Loop over the species positions
            data[counter] = tf.math.reduce_sum(tensor, axis=0)  # Sum over the positions of the atoms
            counter += 1                                        # update the counter
        dipole_moment = tf.convert_to_tensor(data)              # Convert the results to a tf.tensor

        # Build the charge tensor for assignment
        system_charges = [self.parent.species[atom]['charge'][0] for atom in self.parent.species]  # load species charge
        charge_tuple = []              # define empty array for the charges
        for charge in system_charges:  # loop over each species charge
            # Build a tensor of charges allowing for memory management.
            charge_tuple.append(tf.ones([self.batch_size['Parallel'], 3], dtype=tf.float64) * charge)

        charge_tensor = tf.stack(charge_tuple)                # stack the tensors into a single object
        dipole_moment *= charge_tensor                        # Multiply the dipole moment tensor by the system charges
        dipole_moment = tf.reduce_sum(dipole_moment, axis=0)  # Calculate the final dipole moments

        return dipole_moment

    def _calculate_ionic_conductivity(self):
        """
        Calculate the conductivity
        """

        # Calculate the prefactor
        numerator = (self.parent.units['length'] ** 2) * (elementary_charge ** 2)
        denominator = 6 * self.parent.units['time'] * (self.parent.volume * self.parent.units['length'] ** 3) * \
                      self.parent.temperature * boltzmann_constant
        prefactor = numerator / denominator

        dipole_msd_array = self.msd_operation_EH(type_batches='Parallel')

        dipole_msd_array /= int(self.n_batches['Parallel']*self.batch_loop)  # scale by the number of batches

        dipole_msd_array *= prefactor

        popt, pcov = curve_fit(meta_functions.linear_fitting_function, self.time, dipole_msd_array)
        self.parent.ionic_conductivity["Einstein-Helfand"] = [popt[0] / 100, np.sqrt(np.diag(pcov))[0]/100]
        self._update_properties_file(data=[str(popt[0] / 100), str(np.sqrt(np.diag(pcov))[0]/100)])

        # Update the plot if required
        if self.plot:
            plt.plot(np.array(self.time) * self.parent.units['time'], dipole_msd_array)
            self._plot_data()

        # Save the array if required
        if self.save:
            self._save_data(f"{self.analysis_name}", [self.time, dipole_msd_array])

    def run_analysis(self):
        """
        Collect methods and run analysis
        """

        self._autocorrelation_time()            # get the correct correlation time
        self._collect_machine_properties()      # collect machine properties and determine batch size
        self._calculate_batch_loop()            # Update the batch loop attribute
        status = self._check_input()            # Check for bad input
        if status == -1:
            return
        else:
            self._calculate_ionic_conductivity()    # calculate the ionic conductivity

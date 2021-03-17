"""
Class for the calculation of the Green-Kubo ionic conductivity.

Summary
This module contains the code for the Green-Kubo viscsity class. This class is called by the
"""
import os
import warnings

import h5py as hf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# Import user packages
from tqdm import tqdm

from mdsuite.calculators.calculator import Calculator
# Import MDSuite modules
from mdsuite.database.database import Database
from mdsuite.utils.meta_functions import join_path

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboViscosity(Calculator):
    """ Class for the Green-Kubo ionic conductivity implementation

    Attributes
    ----------
    obj :  object
            Experiment class to call from
    plot : bool
            if true, plot the data
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

    def __init__(self, obj, plot=False, data_range=500, x_label='Time (s)', y_label=r'SACF ($C^{2}\cdot m^{2}/s^{2}$)',
                 save=True, analysis_name='green_kubo_viscosity', correlation_time: int = 1):
        """

        Attributes
        ----------
        obj :  object
                Experiment class to call from
        plot : bool
                if true, plot the data
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
        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name,
                         correlation_time=correlation_time, parallel=True)

        self.loaded_property = 'Momentum_Flux'  # property to be loaded for the analysis
        self.tensor_choice = False  # Load data as a tensor
        self.database_group = 'viscosity'  # Which database group to save the data in
        self.loaded_properties = {'Stress'}  # property to be loaded for the analysis

        # Check if current was already computed
        with hf.File(os.path.join(obj.database_path, 'database.hdf5'), "r+") as database:
            # Unwrap the positions if they need to be unwrapped
            if self.loaded_property not in database:
                print(f"Calculating the {self.loaded_property} current")
                self._calculate_system_current()
                print("Current calculation is finished and stored in the database, proceeding with analysis")

    def _autocorrelation_time(self):
        """
        calculate the current autocorrelation time for correct sampling
        """
        pass

    def _calculate_system_current(self):
        """
        Calculate the ionic current of the system

        Parameters
        ----------
        velocity_matrix : np.array
                tensor of system velocities for use in the current calculation

        Returns
        -------
        system_current : np.array
                thermal current of the system as a vector of shape (number_of_configurations, 3)
        """

        # collect machine properties and determine batch size
        self._collect_machine_properties(group_property='Velocities')
        n_batches = np.floor(self.parent.number_of_configurations / self.batch_size['Parallel'])
        remainder = int(self.parent.number_of_configurations % self.batch_size['Parallel'])

        # add a dataset in the database and prepare the structure
        database = Database(name=os.path.join(self.parent.database_path, "database.hdf5"), architecture='simulation')
        db_object = database.open()  # open a database
        path = join_path(self.loaded_property, self.loaded_property)  # name of the new database
        dataset_structure = {path: (self.parent.number_of_configurations, 3)}
        database.add_dataset(dataset_structure, db_object)  # add a new dataset to the database
        data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2]}}

        # process the batches
        for i in tqdm(range(int(n_batches)), ncols=70):
            stress_tensor = self._load_batch(i, "Stress", sym_matrix=True)

            # we take the xy, xz, and yz components (negative)
            phi_x = -stress_tensor[:, :, 3]
            phi_y = -stress_tensor[:, :, 4]
            phi_z = -stress_tensor[:, :, 5]

            phi = np.dstack([phi_x, phi_y, phi_z])

            phi_sum_atoms = phi.sum(axis=0)

            system_current = phi_sum_atoms  # returns the same values as in the compute flux of lammps

            database.add_data(data=system_current,
                              structure=data_structure,
                              database=db_object,
                              start_index=i,
                              batch_size=self.batch_size['Parallel'],
                              system_tensor=True)

        if remainder > 0:
            start = self.parent.number_of_configurations - remainder

            stress_tensor = self.parent.load_matrix('Stress', select_slice=np.s_[:, start:],
                                                    tensor=self.tensor_choice, scalar=False,
                                                    sym_matrix=True)  # Load the stress tensor

            # we take the xy, xz, and yz components (negative)
            phi_x = -stress_tensor[:, :, 3]
            phi_y = -stress_tensor[:, :, 4]
            phi_z = -stress_tensor[:, :, 5]

            phi = np.dstack([phi_x, phi_y, phi_z])

            phi_sum_atoms = phi.sum(axis=0)

            system_current = phi_sum_atoms  # returns the same values as in the compute flux of lammps

            database.add_data(data=system_current,
                              structure=data_structure,
                              database=db_object,
                              start_index=start,
                              batch_size=remainder,
                              system_tensor=True)

        database.close(db_object)  # close the database
        self.parent.memory_requirements = database.get_memory_information()  # update the memory info in experiment

    def _calculate_viscosity(self):
        """
        Calculate the viscosity of the system
        """

        # prepare the prefactor for the integral
        # Since lammps gives the stress in pressure*volume, then we need to add to the denominator volume**2,
        # this is why the numerator becomes 1, and volume appears in the denominator.
        numerator = 1  # self.parent.volume
        denominator = 3 * (self.data_range - 1) * self.parent.temperature * self.parent.units[
            'boltzman'] * self.parent.volume  # we use boltzman constant in the units provided.

        db_path = join_path(self.loaded_property, self.loaded_property)

        prefactor = numerator / denominator

        sigma = []
        parsed_autocorrelation = tf.zeros(self.data_range, dtype=tf.float64)

        for i in range(int(self.n_batches['Parallel'])):  # loop over batches
            batch = self._load_batch(i, path=db_path)

            def generator():
                for start_index in range(int(self.batch_loop)):
                    start = start_index * self.correlation_time
                    stop = start + self.data_range
                    yield batch[start:stop]

            dataset = tf.data.Dataset.from_generator(
                generator,
                output_signature=tf.TensorSpec(shape=(self.data_range, 3), dtype=tf.float64)
            )
            dataset = dataset.map(self.convolution_op, num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                  deterministic=False)

            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

            # for frame in tqdm(dataset.batch(self.data_range), total=int(self.batch_loop / self.data_range)):
            for frame in tqdm(dataset, total=int(self.batch_loop), smoothing=0.1):
                parsed_autocorrelation += frame[self.data_range - 1:]
                sigma.append(np.trapz(frame[self.data_range - 1:], x=self.time))

        # convert to SI units.
        prefactor_units = self.parent.units['pressure'] ** 2 * self.parent.units['length'] ** 3 * self.parent.units[
            'time'] / self.parent.units['energy']
        sigma = prefactor * prefactor_units * np.array(sigma)
        parsed_autocorrelation /= max(parsed_autocorrelation)  # Get the normalized autocorrelation plot data

        self._update_properties_file(data=[str(np.mean(sigma)), str((np.std(sigma) / np.sqrt(len(sigma))))])

        plt.plot(self.time * self.parent.units['time'], parsed_autocorrelation)  # Add a plot

        if self.save:
            self._save_data(f'{self.analysis_name}', [self.time, parsed_autocorrelation])

        if self.plot:
            self._plot_data()  # Plot the data if necessary

    def run_analysis(self):
        """
        call relevant methods and run analysis
        """

        self._autocorrelation_time()  # get the autocorrelation time
        self._collect_machine_properties()  # collect machine properties and determine batch size
        self._calculate_batch_loop()  # Update the batch loop attribute
        status = self._check_input()  # Check for bad input
        if status == -1:
            return
        else:
            self._calculate_viscosity()  # calculate the viscosity

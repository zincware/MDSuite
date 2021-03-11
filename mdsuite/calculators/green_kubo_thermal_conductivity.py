"""
Class for the calculation of the Green-Kubo thermal conductivity.

Summary
-------
This module contains the code for the Green-Kubo thermal conductivity class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.green_kubo_thermal_conductivity method.
The methods in class can then be called by the Experiment.green_kubo_thermal_conductivity method and all necessary
calculations performed.
"""
import os
import warnings

import h5py as hf
# Python standard packages
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# Import user packages
from tqdm import tqdm

from mdsuite.calculators.calculator import Calculator
# Set style preferences, turn off warning, and suppress the duplication of loading bars.
from mdsuite.database.database import Database
from mdsuite.utils.meta_functions import join_path

# Import MDSuite modules

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboThermalConductivity(Calculator):
    """
    Class for the Green-Kubo Thermal conductivity implementation

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
    time : np.array
            Array of the time.
    correlation_time : int
            Correlation time of the property being studied. This is used to ensure ensemble sampling is only performed
            on uncorrelated samples. If this is true, the error extracted form the calculation will be correct.
    """

    def __init__(self, obj, plot=False, data_range=500, x_label='Time (s)', y_label='JACF ($C^{2}\\cdot m^{2}/s^{2}$)',
                 save=True, analysis_name='green_kubo_thermal_conductivity',
                 correlation_time: int = 1):
        """
        Class for the Green-Kubo Thermal conductivity implementation

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
        correlation_time: int
        """
        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name,
                         correlation_time=correlation_time, parallel=True)

        self.loaded_property = 'Thermal_Flux'  # property to be loaded for the analysis
        self.database_group = 'thermal_conductivity'  # Which database group to save the data in
        self.loaded_properties = {'Velocities', 'Stress', 'ke', 'pe'}  # property to be loaded for the analysis
        self.tensor_choice = False  # Load data as a tensor

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
        Calculate the thermal current of the system

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
            velocity_matrix = self._load_batch(i, "Velocities")  # Load the velocity matrix
            stress_tensor = self._load_batch(i, "Stress", sym_matrix=True)
            pe = self._load_batch(i, "PE", scalar=True)
            ke = self._load_batch(i, "KE", scalar=True)

            # define phi as product stress tensor * velocity matrix.
            # It is done by components to take advantage of the symmetric matrix.
            phi_x = np.multiply(stress_tensor[:, :, 0], velocity_matrix[:, :, 0]) + \
                    np.multiply(stress_tensor[:, :, 3], velocity_matrix[:, :, 1]) + \
                    np.multiply(stress_tensor[:, :, 4], velocity_matrix[:, :, 2])
            phi_y = np.multiply(stress_tensor[:, :, 3], velocity_matrix[:, :, 0]) + \
                    np.multiply(stress_tensor[:, :, 1], velocity_matrix[:, :, 1]) + \
                    np.multiply(stress_tensor[:, :, 5], velocity_matrix[:, :, 2])
            phi_z = np.multiply(stress_tensor[:, :, 4], velocity_matrix[:, :, 0]) + \
                    np.multiply(stress_tensor[:, :, 5], velocity_matrix[:, :, 1]) + \
                    np.multiply(stress_tensor[:, :, 2], velocity_matrix[:, :, 2])

            phi = np.dstack([phi_x, phi_y, phi_z])

            phi_sum = phi.sum(axis=0)
            phi_sum_atoms = phi_sum / self.parent.units['NkTV2p']  # factor for units lammps nktv2p

            # ke_total = np.sum(ke, axis=0) # to check it was the same, can be removed.
            # pe_total = np.sum(pe, axis=0)

            energy = ke + pe

            energy_velocity = energy * velocity_matrix
            energy_velocity_atoms = energy_velocity.sum(axis=0)

            system_current = energy_velocity_atoms - phi_sum_atoms  # returns the same values as in the compute flux of lammps

            database.add_data(data=system_current,
                              structure=data_structure,
                              database=db_object,
                              start_index=i,
                              batch_size=self.batch_size['Parallel'],
                              system_tensor=True)

        if remainder > 0:
            start = self.parent.number_of_configurations - remainder
            velocity_matrix = self.parent.load_matrix('Velocities', select_slice=np.s_[:, start:],
                                                      tensor=self.tensor_choice, scalar=False,
                                                      sym_matrix=False)  # Load the velocity matrix
            stress_tensor = self.parent.load_matrix('Stress', select_slice=np.s_[:, start:],
                                                    tensor=self.tensor_choice, scalar=False,
                                                    sym_matrix=True)  # Load the stress tensor

            pe = self.parent.load_matrix('PE', select_slice=np.s_[:, start:],
                                         tensor=self.tensor_choice, scalar=True,
                                         sym_matrix=False)  # Load the potential energy

            ke = self.parent.load_matrix('KE', select_slice=np.s_[:, start:],
                                         tensor=self.tensor_choice, scalar=True,
                                         sym_matrix=False)  # Load the kinetic energy

            # define phi as product stress tensor * velocity matrix.
            # It is done by components to take advantage of the symmetric matrix.
            phi_x = np.multiply(stress_tensor[:, :, 0], velocity_matrix[:, :, 0]) + \
                    np.multiply(stress_tensor[:, :, 3], velocity_matrix[:, :, 1]) + \
                    np.multiply(stress_tensor[:, :, 4], velocity_matrix[:, :, 2])
            phi_y = np.multiply(stress_tensor[:, :, 3], velocity_matrix[:, :, 0]) + \
                    np.multiply(stress_tensor[:, :, 1], velocity_matrix[:, :, 1]) + \
                    np.multiply(stress_tensor[:, :, 5], velocity_matrix[:, :, 2])
            phi_z = np.multiply(stress_tensor[:, :, 4], velocity_matrix[:, :, 0]) + \
                    np.multiply(stress_tensor[:, :, 5], velocity_matrix[:, :, 1]) + \
                    np.multiply(stress_tensor[:, :, 2], velocity_matrix[:, :, 2])

            phi = np.dstack([phi_x, phi_y, phi_z])

            phi_sum_atoms = phi.sum(axis=0) / self.parent.units['NkTV2p']  # factor for units lammps nktv2p

            # ke_total = np.sum(ke, axis=0) # to check it was the same, can be removed.
            # pe_total = np.sum(pe, axis=0)

            energy = ke + pe

            energy_velocity = energy * velocity_matrix
            energy_velocity_atoms = energy_velocity.sum(axis=0)

            system_current = energy_velocity_atoms - phi_sum_atoms  # returns the same values as in the compute flux of lammps

            database.add_data(data=system_current,
                              structure=data_structure,
                              database=db_object,
                              start_index=start,
                              batch_size=remainder,
                              system_tensor=True)

        database.close(db_object)  # close the database
        self.parent.memory_requirements = database.get_memory_information()  # update the memory info in experiment

    def _calculate_thermal_conductivity(self):
        """
        Calculate the thermal conductivity in the system
        """

        # prepare the prefactor for the integral
        numerator = 1
        denominator = 3 * (self.data_range - 1) * self.parent.temperature ** 2 * self.parent.units['boltzman'] \
                      * self.parent.volume  # we use boltzman constant in the units provided.

        prefactor = numerator / denominator

        db_path = join_path(self.loaded_property, self.loaded_property)

        sigma = []
        parsed_autocorrelation = tf.zeros(self.data_range, dtype=tf.float64)

        for i in range(int(self.n_batches['Parallel'])):  # loop over batches
            batch = self._load_batch(i, path=db_path)

            def generator():
                for start_index in range(int(self.batch_loop)):
                    start = start_index + self.correlation_time
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
        prefactor_units = self.parent.units['energy'] / self.parent.units['length'] / self.parent.units['time']
        sigma = prefactor * prefactor_units * np.array(sigma)
        parsed_autocorrelation /= max(parsed_autocorrelation)  # Get the normalized autocorrelation plot data

        self._update_properties_file(data=[str(np.mean(sigma)), str((np.std(sigma) / np.sqrt(len(sigma))))])

        plt.plot(self.time * self.parent.units['time'], parsed_autocorrelation)  # Add a plot

        if self.save:
            self._save_data(f'{self.analysis_name}', [self.time, parsed_autocorrelation])

        if self.plot:
            self._plot_data()  # Plot the data if necessary

    def run_analysis(self):
        """ Run thermal conductivity calculation analysis

        The thermal conductivity is computed at this step.
        """
        self._autocorrelation_time()  # get the autocorrelation time
        self._collect_machine_properties()  # collect machine properties and determine batch size
        self._calculate_batch_loop()  # Update the batch loop attribute
        status = self._check_input()  # Check for bad input
      
        if status == -1:
            return
        else:
            self._calculate_thermal_conductivity()  # calculate the singular diffusion coefficients

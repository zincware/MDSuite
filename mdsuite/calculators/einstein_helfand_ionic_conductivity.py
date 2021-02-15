"""
Class for the calculation of the Einstein-Helfand ionic conductivity.

Summary
-------
This class is called by the Experiment class and instantiated when the user calls the
Experiment.einstein_helfand_ionic_conductivity method. The methods in class can then be called by the
Experiment.einstein_helfand_ionic_conductivity method and all necessary calculations performed.
"""

import matplotlib
import os
import warnings

import h5py as hf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.optimize import curve_fit

# Import user packages
from tqdm import tqdm

# Import MDSuite modules
import mdsuite.utils.meta_functions as meta_functions
from mdsuite.calculators.calculator import Calculator
from mdsuite.utils.units import elementary_charge, boltzmann_constant
from mdsuite.database.database import Database

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")
matplotlib.use('Agg')


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

        # parse to the parent class
        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name, parallel=True)

        self.loaded_property = 'Translational_Dipole_Moment'  # Property to be loaded for the analysis
        self.batch_loop = None  # Number of ensembles in a batch
        self.tensor_choice = True  # Load data as a tensor
        self.correlation_time = 1  # Correlation time of the current
        self.species = list(obj.species)  # species on which to perform the analysis

        self.database_group = 'ionic_conductivity'  # Which database group to save the data in

        # Time array for the calculations
        self.time = np.linspace(0.0, self.data_range * self.parent.time_step * self.parent.sample_rate, self.data_range)

        with hf.File(os.path.join(obj.database_path, 'database.hdf5'), "r+") as database:
            # Check for unwrapped positions
            for item in self.species:
                # Unwrap the positions if they need to be unwrapped
                if "Unwrapped_Positions" not in database[item]:
                    print("Unwrapping coordinates")
                    obj.perform_transformation('UnwrapCoordinates', species=[item])  # Unwrap the coordinates
                    print("Coordinate unwrapping finished, proceeding with analysis")
                # Check for translational dipole moment
                if self.loaded_property not in database:
                    print("Calculating the translational dipole moment")
                    self._calculate_integrated_current()
                    print("Dipole moment calculation is finished and stored in the database, proceeding with analysis")

    def _autocorrelation_time(self):
        """
        Calculate dipole moment autocorrelation time

        When performing this analysis, the sampling should occur over the autocorrelation time of the positions in the
        system. This method will calculate what this time is and sample over it to ensure uncorrelated samples.
        """
        pass

    def _calculate_integrated_current(self):
        """
        Calculate the translational dipole of the system

        This method will calculate the translational dipole moment of a system and store the data in the database

        Returns
        -------
        Adds the translational dipole moment to the simulation database
        """

        self._collect_machine_properties(group_property='Unwrapped_Positions')
        n_batches = np.floor(self.parent.number_of_configurations / self.batch_size['Parallel'])
        remainder = int(self.parent.number_of_configurations % self.batch_size['Parallel'])

        # add a dataset in the database and prepare the structure
        database = Database(name=os.path.join(self.parent.database_path, "database.hdf5"), architecture='simulation')
        db_object = database.open()  # open a database
        path = os.path.join('Translational_Dipole_Moment', 'Translational_Dipole_Moment')  # name of the new database
        dataset_structure = {path: (self.parent.number_of_configurations, 3)}
        database.add_dataset(dataset_structure, db_object)  # add a new dataset to the database
        data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2]}}

        for i in tqdm(range(int(n_batches)), ncols=70):
            data = self._load_batch(i, loaded_property='Unwrapped_Positions')  # Load the velocity matrix
            counter = 0  # set a counter variable
            for tensor in data:  # Loop over the species positions
                data[counter] = tf.math.reduce_sum(tensor, axis=0)  # Sum over the positions of the atoms
                counter += 1  # update the counter
            dipole_moment = tf.convert_to_tensor(data)  # Convert the results to a tf.tensor

            # Build the charge tensor for assignment
            system_charges = [self.parent.species[atom]['charge'][0] for atom in
                              self.parent.species]  # load species charge
            charge_tuple = []  # define empty array for the charges
            for charge in system_charges:  # loop over each species charge
                # Build a tensor of charges allowing for memory management.
                charge_tuple.append(tf.ones([self.batch_size['Parallel'], 3], dtype=tf.float64) * charge)

            charge_tensor = tf.stack(charge_tuple)  # stack the tensors into a single object
            dipole_moment *= charge_tensor  # Multiply the dipole moment tensor by the system charges
            dipole_moment = tf.reduce_sum(dipole_moment, axis=0)  # Calculate the final dipole moments

            database.add_data(data=dipole_moment,
                              structure=data_structure,
                              database=db_object,
                              start_index=i,
                              batch_size=self.batch_size['Parallel'],
                              system_tensor=True)
            # fetch remainder if worth while
            if remainder > 0:
                start = self.parent.number_of_configurations - remainder
                data = self.parent.load_matrix('Unwrapped_Positions', select_slice=np.s_[:, start:],
                                               tensor=self.tensor_choice, scalar=False, sym_matrix=False)

                counter = 0  # set a counter variable
                for tensor in data:  # Loop over the species positions
                    data[counter] = tf.math.reduce_sum(tensor, axis=0)  # Sum over the positions of the atoms
                    counter += 1  # update the counter
                dipole_moment = tf.convert_to_tensor(data)  # Convert the results to a tf.tensor

                # Build the charge tensor for assignment
                system_charges = [self.parent.species[atom]['charge'][0] for atom in
                                  self.parent.species]  # load species charge
                charge_tuple = []  # define empty array for the charges
                for charge in system_charges:  # loop over each species charge
                    # Build a tensor of charges allowing for memory management.
                    charge_tuple.append(tf.ones([self.batch_size['Parallel'], 3], dtype=tf.float64) * charge)

                charge_tensor = tf.stack(charge_tuple)  # stack the tensors into a single object
                dipole_moment *= charge_tensor  # Multiply the dipole moment tensor by the system charges
                dipole_moment = tf.reduce_sum(dipole_moment, axis=0)  # Calculate the final dipole moments

                database.add_data(data=dipole_moment,
                                  structure=data_structure,
                                  database=db_object,
                                  start_index=i,
                                  batch_size=self.batch_size['Parallel'],
                                  system_tensor=True)
            database.close(db_object)  # close the database
            self.parent.memory_requirements = database.get_memory_information()  # update the memory info in experiment

    def _calculate_ionic_conductivity(self):
        """
        Calculate the conductivity
        """

        # Calculate the prefactor
        numerator = (self.parent.units['length'] ** 2) * (elementary_charge ** 2)
        denominator = 6 * self.parent.units['time'] * (self.parent.volume * self.parent.units['length'] ** 3) * \
                      self.parent.temperature * boltzmann_constant
        prefactor = numerator / denominator

        group = os.path.join(self.loaded_property, self.loaded_property)
        dipole_msd_array = self.msd_operation_EH(group=group)
        dipole_msd_array /= int(self.n_batches['Parallel'] * self.batch_loop)  # scale by the number of batches
        dipole_msd_array *= prefactor
        popt, pcov = curve_fit(meta_functions.linear_fitting_function, self.time, dipole_msd_array)
        self._update_properties_file(data=[str(popt[0] / 100), str(np.sqrt(np.diag(pcov))[0] / 100)])

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

        self._autocorrelation_time()  # get the correct correlation time
        self._collect_machine_properties()  # collect machine properties and determine batch size
        self._calculate_batch_loop()  # Update the batch loop attribute
        status = self._check_input()  # Check for bad input
        if status == -1:
            return
        else:
            self._calculate_ionic_conductivity()  # calculate the ionic conductivity

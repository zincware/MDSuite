"""
Class for the calculation of the Green-Kubo ionic conductivity.

Summary
This module contains the code for the Green-Kubo ionic conductivity class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.green_kubo_ionic_conductivity method.
The methods in class can then be called by the Experiment.green_kubo_ionic_conductivity method and all necessary
calculations performed.
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import warnings

# Import user packages
from tqdm import tqdm
import h5py as hf

# Import MDSuite modules
from mdsuite.utils.units import boltzmann_constant, elementary_charge
from mdsuite.database.database import Database
from mdsuite.calculators.calculator import Calculator

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboIonicConductivity(Calculator):
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

    def __init__(self, obj, plot=False, data_range=500, x_label='Time (s)', y_label=r'JACF ($C^{2}\cdot m^{2}/s^{2}$)',
                 save=True, analysis_name='green_kubo_ionic_conductivity', correlation_time=1):
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

        # update parent class
        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name, parallel=True,
                         correlation_time=correlation_time)

        self.loaded_property = 'Ionic_Current'  # property to be loaded for the analysis
        self.batch_loop = None  # Number of ensembles in each batch
        self.tensor_choice = False  # Load data as a tensor
        self.database_group = 'ionic_conductivity'  # Which database group to save the data in
        self.time = np.linspace(0.0, data_range * self.parent.time_step * self.parent.sample_rate, data_range)
        self.correlation_time = 1  # correlation time of the system current.

        # Check for unwrapped coordinates and unwrap if not stored already.
        with hf.File(os.path.join(obj.database_path, 'database.hdf5'), "r+") as database:
            # Unwrap the positions if they need to be unwrapped
            if self.loaded_property not in database:
                print("Calculating the ionic current")
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

        Returns
        -------
        Updates the simulation database with the ionic current property
        """

        # collect machine properties and determine batch size
        self._collect_machine_properties(group_property='Velocities')
        n_batches = np.floor(self.parent.number_of_configurations / self.batch_size['Parallel'])
        remainder = int(self.parent.number_of_configurations % self.batch_size['Parallel'])

        # add a dataset in the database and prepare the structure
        database = Database(name=os.path.join(self.parent.database_path, "database.hdf5"), architecture='simulation')
        db_object = database.open()  # open a database
        path = os.path.join('Ionic_Current', 'Ionic_Current')  # name of the new database
        dataset_structure = {path: (self.parent.number_of_configurations, 3)}
        database.add_dataset(dataset_structure, db_object)  # add a new dataset to the database
        data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2]}}

        # process the batches
        for i in tqdm(range(int(n_batches)), ncols=70):
            velocity_matrix = self._load_batch(i, loaded_property='Velocities')  # load a batch of data
            # build charge array
            species_charges = [self.parent.species[atom]['charge'][0] for atom in self.parent.species]

            system_current = np.zeros((self.batch_size['Parallel'], 3))  # instantiate the current array
            # Calculate the total system current
            for j in range(len(velocity_matrix)):
                system_current += np.array(np.sum(velocity_matrix[j][:, 0:], axis=0)) * species_charges[j]

            database.add_data(data=system_current,
                              structure=data_structure,
                              database=db_object,
                              start_index=i,
                              batch_size=self.batch_size['Parallel'],
                              system_tensor=True)

        if remainder > 0:
            start = self.parent.number_of_configurations - remainder
            velocity_matrix = self.parent.load_matrix('Velocities', select_slice=np.s_[:, start:],
                                                      tensor=self.tensor_choice, scalar=False, sym_matrix=False)
            # build charge array
            species_charges = [self.parent.species[atom]['charge'][0] for atom in self.parent.species]

            system_current = np.zeros((len(velocity_matrix), 3))  # instantiate the current array
            # Calculate the total system current
            for j in range(len(velocity_matrix)):
                system_current += np.array(np.sum(velocity_matrix[j][:, 0:], axis=0)) * species_charges[j]

            database.add_data(data=system_current,
                              structure=data_structure,
                              database=db_object,
                              start_index=start,
                              batch_size=remainder,
                              system_tensor=True)
        database.close(db_object)  # close the database
        self.parent.memory_requirements = database.get_memory_information()  # update the memory info in experiment

    def _calculate_ionic_conductivity(self):
        """
        Calculate the ionic conductivity in the system
        """

        # Calculate the prefactor
        numerator = (elementary_charge ** 2) * (self.parent.units['length'] ** 2)
        denominator = 3 * boltzmann_constant * self.parent.temperature * self.parent.volume * \
                      (self.parent.units['length'] ** 3) * self.data_range * self.parent.units['time']
        prefactor = numerator / denominator

        db_path = os.path.join(self.loaded_property, self.loaded_property)
        sigma, parsed_autocorrelation = self.convolution_operation(group=db_path)
        sigma *= prefactor

        # update the experiment class
        self._update_properties_file(data=[str(np.mean(sigma) / 100), str((np.std(sigma) / np.sqrt(len(sigma))) / 100)])

        plt.plot(self.time * self.parent.units['time'], parsed_autocorrelation)  # Add a plot

        parsed_autocorrelation /= max(parsed_autocorrelation)  # Get the normalized autocorrelation plot data

        # save the data if necessary
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
            self._calculate_ionic_conductivity()  # calculate the ionic conductivity

"""
Class for the calculation of the Einstein-Helfand ionic conductivity.

Summary
-------
This class is called by the Experiment class and instantiated when the user calls the
Experiment.einstein_helfand_ionic_conductivity method. The methods in class can then be called by the
Experiment.einstein_helfand_ionic_conductivity method and all necessary calculations performed.
"""

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
# Set style preferences, turn off warning, and suppress the duplication of loading bars.
from mdsuite.database.database import Database

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class EinsteinHelfandThermalKinaci(Calculator):
    """
    Class for the Einstein-Helfand Ionic Conductivity

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

    def __init__(self, obj, plot=True, data_range=500, save=True,
                 x_label='Time (s)', y_label='MSD (m$^2$/s)', analysis_name='einstein_helfand_thermal_conductivity_kinaci',
                 correlation_time=1):
        """
        Python constructor

        Parameters
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

        # parse to the parent class
        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name, parallel=True,
                         correlation_time=correlation_time)

        self.loaded_property = 'Integrated_heat_current_kinaci'  # Property to be loaded for the analysis
        self.batch_loop = None  # Number of ensembles in a batch
        self.parallel = True  # Set the parallel attribute
        self.tensor_choice = True  # Load data as a tensor
        self.species = list(obj.species)  # species on which to perform the analysis

        self.database_group = 'thermal_conductivity'  # Which database group to save the data in

        # Time array for the calculations
        self.time = np.linspace(0.0, self.data_range * self.parent.time_step * self.parent.sample_rate, self.data_range)

        # Check for unwrapped coordinates and unwrap if not stored already.

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
                    print("Calculating integrated heat current")
                    self._calculate_integrated_current()
                    print(
                        "Integrated heat current computation is finished and stored in the database, proceeding with analysis")

    def _autocorrelation_time(self):
        """
        Calculate autocorrelation time

        When performing this analysis, the sampling should occur over the autocorrelation time of the positions in the
        system. This method will calculate what this time is and sample over it to ensure uncorrelated samples.
        """
        pass

    def _calculate_integrated_current(self):
        """
        Calculate the integrated heat current of the system

        Parameters
        ----------
        data : list
                A list of tensor corresponding to the positions of the particles.

        Returns
        -------
        dipole_moment : tf.tensor
                Return the dipole moment for the batch
        """

        # add a dataset in the database and prepare the structure
        database = Database(name=os.path.join(self.parent.database_path, "database.hdf5"), architecture='simulation')
        db_object = database.open()  # open a database
        path = meta_functions.join_path(self.loaded_property, self.loaded_property)  # name of the new database
        dataset_structure = {path: (self.parent.number_of_configurations, 3)}
        database.add_dataset(dataset_structure, db_object)  # add a new dataset to the database
        data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2]}}

        positions = self.parent.load_matrix("Unwrapped_Positions", select_slice=np.s_[:],
                                tensor=self.tensor_choice)

        velocities = self.parent.load_matrix("Velocities", select_slice=np.s_[:],
                                tensor=self.tensor_choice)

        forces = self.parent.load_matrix("Forces", select_slice=np.s_[:],
                                             tensor=self.tensor_choice)

        PE = self.parent.load_matrix("PE", select_slice=np.s_[:],
                                             tensor=self.tensor_choice)

        PE = tf.squeeze(PE)

        integrand = tf.einsum('ijk,ijk->ij', forces, velocities)
        integral = tf.cumsum(integrand, axis=1) * self.parent.time_step * self.parent.sample_rate
        R_K = tf.einsum('ijk,ij->jk', positions, integral)

        R_P = tf.einsum('ij,ijk->jk', PE, positions) # does not contribute for solids

        integrated_heat_current = R_K + R_P

        database.add_data(data=integrated_heat_current,
                          structure=data_structure,
                          database=db_object,
                          start_index=0,
                          batch_size=self.parent.number_of_configurations,
                          system_tensor=True)


        database.close(db_object)  # close the database
        self.parent.memory_requirements = database.get_memory_information()  # update the memory info in experiment

    def _calculate_thermal_conductivity(self):
        """
        Calculate the conductivity
        """

        # Calculate the prefactor
        numerator = 1
        denominator = 2 * self.parent.volume * self.parent.temperature * self.parent.units['boltzman']
        units_change = self.parent.units['energy'] / self.parent.units['length'] / self.parent.units['time'] / \
                       self.parent.units['temperature']
        prefactor = numerator / denominator * units_change

        group = meta_functions.join_path(self.loaded_property, self.loaded_property)
        msd_array = self.msd_operation_EH(group=group)
        msd_array /= int(self.n_batches['Parallel'] * self.batch_loop)  # scale by the number of batches
        msd_array *= prefactor
        popt, pcov = curve_fit(meta_functions.linear_fitting_function, self.time, msd_array)
        self._update_properties_file(data=[str(popt[0]), str(np.sqrt(np.diag(pcov))[0])])

        # Update the plot if required
        if self.plot:
            plt.plot(np.array(self.time) * self.parent.units['time'], msd_array)
            self._plot_data()

        # Save the array if required
        if self.save:
            self._save_data(f"{self.analysis_name}", [self.time, msd_array])

    def run_analysis(self):
        """
        Collect methods and run analysis
        """

        self._autocorrelation_time()  # get the correct correlation time
        self.collect_machine_properties()  # collect machine properties and determine batch size
        self._calculate_batch_loop()  # Update the batch loop attribute
        status = self._check_input()  # Check for bad input
        if status == -1:
            return
        else:
            self._calculate_thermal_conductivity()  # calculate the ionic conductivity

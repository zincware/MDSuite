"""
Parent class for different analysis

Summary
-------
"""

import abc
import random

from scipy.optimize import curve_fit
import yaml
import h5py as hf
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import signal
from scipy.optimize import curve_fit
from tqdm import tqdm

from mdsuite.utils.exceptions import *
from mdsuite.utils.meta_functions import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mdsuite.experiment.experiment import Experiment


class Calculator(metaclass=abc.ABCMeta):
    """
    Parent class for analysis modules

    Attributes
    ----------
    obj : class object
            Class object of the experiment.
    plot : bool (default=True)
            Decision to plot the analysis.
    save : bool (default=True)
            Decision to save the generated data arrays.

    data_range : int (default=500)
            Range over which the property should be evaluated. This is not applicable to the current
            analysis as the full rdf will be calculated.
    x_label : str
            How to label the x axis of the saved plot.
    y_label : str
            How to label the y axis of the saved plot.
    analysis_name : str
            Name of the analysis. used in saving of the data and figure.
    batch_size : dict
            Size of batches to use in the analysis separated into parallel and serial components, i.e
            {'Serial': 100, 'Parallel': 50} for a two component, symmetric system.
    n_batches : dict
            Number of barthes to use as a dictionary for both serial and parallel implementations
    machine_properties : dict
            Devices available to MDSuite during analysis run. Has the following structure
            {'memory': x bytes, 'cpus': n_of_cores, 'gpus': name_of_gpu}

    """

    def __init__(self, obj: "Experiment", plot=True, save=True, data_range=500, x_label=None, y_label=None, analysis_name=None,
                 parallel=False, correlation_time=1, optimize_correlation_time=False):
        """

        Parameters
        ----------
        obj
        plot
        save
        data_range
        x_label
        y_label
        analysis_name

        """

        self.parent = obj  # Experiment object to get properties from
        self.data_range = data_range  # Data range over which to evaluate
        self.plot = plot  # Whether or not to plot the data and save a figure
        self.save = save  # Whether or not to save the calculated data (Default is true)
        self.loaded_property = None  # Which dataset to load
        self.parallel = parallel  # If true, run analysis in parallel
        self.batch_type = None  # Choose parallel or serial for memory management
        self.tensor_choice = False  # If true, data is loaded as a tensors
        self.batch_size = {}  # Size of the batch to use during the analysis
        self.n_batches = {}  # Number of batches to be calculated over
        self.machine_properties = None  # dictionary of machine properties to be evaluated at analysis run-time
        self.correlation_time = correlation_time  # correlation time of the property

        self.x_label = x_label  # x label of the figure
        self.y_label = y_label  # y label of the figure
        self.analysis_name = analysis_name  # what to save the figure as

        self.database_group = None  # Which database group to save the data in

        # Solve for the batch type
        if self.parallel:
            self.batch_type = 'Parallel'
        else:
            self.batch_type = 'Serial'

        if optimize_correlation_time:
            print("Sorry, this feature is not currently available, please set the correlation time manually.")

    def _autocorrelation_time(self):
        """
        get the autocorrelation time for the relevant property to ensure good error sampling
        """
        raise NotImplementedError  # Implemented in the child class

    def _collect_machine_properties(self, scaling_factor: int = 1, group_property: str = None):
        """
        Collect properties of machine being used.

        This method will collect the properties of the machine being used and parse them to the relevant analysis in
        order to optimize the property computation.

        Parameters
        ----------
        group_property : str
                Property for which memory information should be collected.
        scaling_factor : int
                Amount by which an analysis will expand the dataset.
        """
        if group_property is None:
            group_property = self.loaded_property

        self.machine_properties = get_machine_properties()  # load the machine properties
        memory_usage = []
        for item in self.parent.memory_requirements:
            if group_property in item:
                memory_usage.append(self.parent.memory_requirements[item] / self.parent.number_of_configurations)

        # Get the single frame memory usage in bytes
        serial_memory_usage = scaling_factor*max(memory_usage)
        parallel_memory_usage = scaling_factor*sum(memory_usage)

        # Update the batch_size attribute
        max_batch_size_serial = int(np.floor(0.1*self.machine_properties['memory'] / serial_memory_usage))
        max_batch_size_parallel = int(np.floor(0.1*self.machine_properties['memory'] / parallel_memory_usage))

        if max_batch_size_serial > self.parent.number_of_configurations:
            self.batch_size['Serial'] = self.parent.number_of_configurations

        else:
            self.batch_size['Serial'] = max_batch_size_serial
        
        self.n_batches['Serial'] = np.ceil(self.parent.number_of_configurations /
                                           (self.batch_size['Serial'])).astype(int)

        if max_batch_size_parallel > self.parent.number_of_configurations:
            self.batch_size['Parallel'] = self.parent.number_of_configurations
        else:
            self.batch_size['Parallel'] = max_batch_size_parallel

        self.n_batches['Parallel'] = np.ceil(self.parent.number_of_configurations /
                                             (self.batch_size['Parallel'])).astype(int)

    def _calculate_batch_loop(self):
        """
        Calculate the batch loop parameters
        """
        self.batch_loop = np.floor(
            (self.batch_size[self.batch_type] - self.data_range) / (self.correlation_time + 1)) + 1

    def _load_batch(self, batch_number, loaded_property=None, item=None, scalar=False, sym_matrix=False, path=None):
        """
        Load a batch of data

        Parameters
        ----------
        batch_number : int
                Which batch is being studied
        item : str
                Species being studied at the time

        Returns
        -------
        data array : np.array, tf.tensor
                This implementation returns a tensor of the species positions.
        """
        start = int(batch_number * self.batch_size[self.batch_type])
        stop = int(start + self.batch_size[self.batch_type])

        if loaded_property is None:
            loaded_property = self.loaded_property

        return self.parent.load_matrix(loaded_property, species=item, select_slice=np.s_[:, start:stop],
                                       tensor=self.tensor_choice, scalar=scalar, sym_matrix=sym_matrix, path=path)

    def _save_data(self, title, data):
        """
        Save data to the save data directory

        Parameters
        ----------
        title : str
                name of the data to save. Usually this is just the analysis name. In the case of species specific
                analysis, this will be further appended to include the name of the species.
        data : np.array
                Data to be saved.
        """

        with hf.File(os.path.join(self.parent.database_path, 'analysis_data.hdf5'), 'r+') as db:
            if title in db[self.database_group].keys():
                del db[self.database_group][title]
                db[self.database_group].create_dataset(title, data=data, dtype=float)
            else:
                db[self.database_group].create_dataset(title, data=data, dtype=float)

    def _plot_data(self, title=None, manual=False):
        """
        Plot the data generated during the analysis
        """

        if title is None:
            title = f"{self.analysis_name}"

        if manual:
            plt.savefig(os.path.join(self.parent.figures_path, f"{title}.svg"), dpi=600, format='svg')
        else:
            plt.xlabel(rf'{self.x_label}')  # set the x label
            plt.ylabel(rf'{self.y_label}')  # set the y label
            plt.legend()  # enable the legend
            plt.savefig(os.path.join(self.parent.figures_path, f"{title}.svg"), dpi=600, format='svg')

    def _perform_garbage_collection(self):
        """
        Perform garbage collection after an analysis
        """
        raise NotImplementedError

    def _check_input(self):
        """
        Look for user input that would kill the analysis

        Returns
        -------
        status: int
            if 0, check failed, if 1, check passed.
        """
        if self.data_range > self.parent.number_of_configurations - self.correlation_time:
            print("Data range is impossible for this system, reduce and try again")

            return -1
        else:
            return 0

    def _optimize_einstein_data_range(self, data):
        """
        Optimize the data range of a system using the Einstein method of calculation.

        Parameters
        ----------
        data : np.array (2, data_range)
                MSD to study

        Returns
        -------
        Updates the data_range attribute of the class state
        """

        def func(x, m, a):
            """
            Standard linear function for fitting.

            Parameters
            ----------
            x : list/np.array
                    x axis data for the fit
            m : float
                    gradient of the line
            a : float
                    scalar offset, also the y-intercept for those who did not get much maths in school.

            Returns
            -------

            """

            return m * x + a

        # get the logarithmic dataset
        log_y = np.log10(data[0])
        log_x = np.log10(data[1])

        end_index = int(len(log_y) - 1)
        start_index = int(0.4 * len(log_y))

        popt, pcov = curve_fit(func, log_x[start_index:end_index], log_y[start_index:end_index])  # fit linear regime

        if 0.85 < popt[0] < 1.15:
            self.loop_condition = True

        else:
            try:
                self.data_range = int(1.1 * self.data_range)
                self.time = np.linspace(0.0, self.data_range * self.parent.time_step * self.parent.sample_rate,
                                        self.data_range)
                # end the calculation if the data range exceeds the relevant bounds
                if self.data_range > self.parent.number_of_configurations - self.correlation_time:
                    print("Trajectory not long enough to perform analysis.")
                    raise RangeExceeded
            except RangeExceeded:
                raise RangeExceeded

    @staticmethod
    def _fit_einstein_curve(data):
        """
        Fit operation for Einstein calculations

        Parameters
        ----------
        data : list
                x and y data for the fitting [np.array, np.array] of (2, data_range)

        Returns
        -------
        fit results : list
                A tuple list with the fit value along with the error of the fit
        """

        fits = []  # define an empty fit array so errors may be extracted

        def func(x, m, a):
            """
            Standard linear function for fitting.

            Parameters
            ----------
            x : list/np.array
                    x axis data for the fit
            m : float
                    gradient of the line
            a : float
                    scalar offset, also the y-intercept for those who did not get much maths in school.

            Returns
            -------

            """
            return m * x + a

        # get the logarithmic dataset
        log_y = np.log10(data[1][1:])
        log_x = np.log10(data[0][1:])

        min_end_index, max_end_index = int(0.8 * len(log_y)), int(len(log_y) - 1)
        min_start_index, max_start_index = int(0.3 * len(log_y)), int(0.5 * len(log_y))


        for _ in range(100):
            end_index = random.randint(min_end_index, max_end_index)  # get a random end point
            start_index = random.randint(min_start_index, max_start_index)  # get a random start point

            popt, pcov = curve_fit(func, log_x[start_index:end_index], log_y[start_index:end_index])  # fit linear func
            fits.append(10 ** popt[1])

        return [str(np.mean(fits)), str(np.std(fits))]

    def _update_properties_file(self, item=None, sub_item=None, data=None, add=False):
        """
        Update the system properties YAML file.
        """

        # Check if data has been given
        if data is None:
            print("No data provided")
            return

        with open(os.path.join(self.parent.database_path, 'system_properties.yaml')) as pfr:
            properties = yaml.load(pfr, Loader=yaml.Loader)  # collect the data in the yaml file

        with open(os.path.join(self.parent.database_path, 'system_properties.yaml'), 'w') as pfw:
            if item is None:
                properties[self.database_group][self.analysis_name] = data
            elif sub_item is None:
                properties[self.database_group][self.analysis_name][item] = data
            else:
                if add:
                    properties[self.database_group][self.analysis_name][item] = {}
                    properties[self.database_group][self.analysis_name][item][sub_item] = data
                else:
                    properties[self.database_group][self.analysis_name][item][sub_item] = data

            yaml.dump(properties, pfw)

    @abc.abstractmethod
    def run_analysis(self):
        """
        Run the appropriate analysis
        Should follow the general outline detailed below:
        self._autocorrelation_time()  # Calculate the relevant autocorrelation time
        self._analysis()  # Can be diffusion coefficients or whatever is being calculated, but run the calculation
        self._error_analysis  # Run an error analysis, could be done during the calculation.
        self._update_experiment  # Update the main experiment class with the calculated properties

        """
        raise NotImplementedError  # Implement in the child class

    def convolution_operation(self, group: str = None):
        """
        This function performs the actual autocorrelation computation.
        It is has been put here because it is the same function for every GK calculation.

        :param type_batches: Serial or Parallel.
        :return: sigma: list with the integrated property.
        :return: parsed_autocorrelation: np array with the sum of the autocorrelations, used to see convergence.

        Parameters
        ----------
        group
        """
        sigma = []  # define an empty sigma list
        parsed_autocorrelation = np.zeros(self.data_range)  # Define the parsed array

        for i in tqdm(range(int(self.n_batches['Parallel'])), ncols=70):  # loop over batches
            batch = self._load_batch(i, path=group)  # get the ionic current batch
            for start_index in range(int(self.batch_loop)):  # loop over ensembles in batch
                start = int(start_index + self.correlation_time)  # get start index
                stop = int(start + self.data_range)  # get stop index
                system_current = np.array(batch, dtype=float)[start:stop]  # load data from batch array

                jacf = np.zeros(2 * self.data_range - 1)  # Define the empty jacf array

                # Calculate the current autocorrelation
                jacf += (signal.correlate(system_current[:, 0],
                                          system_current[:, 0],
                                          mode='full', method='auto') +
                         signal.correlate(system_current[:, 1],
                                          system_current[:, 1],
                                          mode='full', method='auto') +
                         signal.correlate(system_current[:, 2],
                                          system_current[:, 2],
                                          mode='full', method='auto'))

                jacf = jacf[int((len(jacf) / 2)):]  # Cut the negative part of the current autocorrelation
                parsed_autocorrelation += jacf  # update parsed function
                sigma.append(np.trapz(jacf, x=self.time))  # Update the conductivity array
        return np.array(sigma), parsed_autocorrelation

    def msd_operation_EH(self, group: str = None):
        """
        A function that needs a docstring
        Parameters
        ----------
        group
        type_batches

        Returns
        -------

        """
        msd_array = np.zeros(self.data_range)  # Initialize the msd array

        for i in tqdm(range(int(self.n_batches[self.batch_type])), ncols=70):  # Loop over batches
            batch = self._load_batch(i, path=group)  # get the ionic current
            for start_index in range(int(self.batch_loop)):  # Loop over ensembles
                start = int(start_index + self.correlation_time)  # get start configuration
                stop = int(start + self.data_range)  # get the stop configuration
                window_tensor = batch[start:stop]  # select data from the batch tensor

                # Calculate the msd and multiply by the prefactor
                msd = (window_tensor - (
                    tf.repeat(tf.expand_dims(window_tensor[0], 0), self.data_range, axis=0))) ** 2
                msd = tf.reduce_sum(msd, axis=1)

                msd_array += np.array(msd)  # Update the total array
        return msd_array

    def _calculate_system_current(self, i):
        pass

"""
Parent class for different analysis

Summary
-------
"""

import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit
import yaml
import h5py as hf

from mdsuite.utils.meta_functions import *
from mdsuite.utils.exceptions import *

import abc


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

    def __init__(self, obj, plot=True, save=True, data_range=500, x_label=None, y_label=None, analysis_name=None):
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
        self.parallel = False  # If true, run analysis in parallel
        self.batch_type = None  # Choose parallel or serial for memory management
        self.tensor_choice = False  # If true, data is loaded as a tensors
        self.batch_size = {}  # Size of the batch to use during the analysis
        self.n_batches = {}  # Number of batches to be calculated over
        self.machine_properties = None  # dictionary of machine properties to be evaluated at analysis run-time
        self.correlation_time = None  # correlation time of the property

        self.x_label = x_label  # x label of the figure
        self.y_label = y_label  # y label of the figure
        self.analysis_name = analysis_name  # what to save the figure as

        self.database_group = None  # Which database group to save the data in

        # Solve for the batch type
        if self.parallel:
            self.batch_type = 'Parallel'
        else:
            self.batch_type = 'Serial'

    def _autocorrelation_time(self):
        """
        get the autocorrelation time for the relevant property to ensure good error sampling
        """
        raise NotImplementedError  # Implemented in the child class

    def _collect_machine_properties(self):
        """
        Collect properties of machine being used.

        This method will collect the properties of the machine being used and parse them to the relevant analysis in
        order to optimize the property computation.
        """

        self.machine_properties = get_machine_properties()  # load the machine properties
        memory_usage = []
        for item in self.parent.memory_requirements:
            memory_usage.append(self.parent.memory_requirements[item][self.loaded_property] /
                                self.parent.number_of_configurations)

        # Get the single frame memory usage in bytes
        serial_memory_usage = max(memory_usage)
        parallel_memory_usage = sum(memory_usage)
        parallel_members = len(memory_usage)

        # Update the batch_size attribute
        max_batch_size_serial = int(np.floor(self.machine_properties['memory'] / serial_memory_usage))
        max_batch_size_parallel = int(np.floor(self.machine_properties['memory'] / parallel_memory_usage))

        self.n_batches['Serial'] = np.ceil(self.parent.number_of_configurations/max_batch_size_serial).astype(int)
        self.batch_size['Serial'] = np.ceil(self.parent.number_of_configurations/self.n_batches['Serial']).astype(int)

        self.n_batches['Parallel'] = np.ceil(self.parent.number_of_configurations / max_batch_size_serial).astype(int)
        self.batch_size['Parallel'] = np.ceil(self.parent.number_of_configurations / self.n_batches['Serial']).astype(int)

    def _calculate_batch_loop(self):
        """
        Calculate the batch loop parameters
        """
        self.batch_loop = np.floor((self.batch_size[self.batch_type] - self.data_range)/(self.correlation_time + 1)) + 1

    def _load_batch(self, batch_number, item=None):
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

        return self.parent.load_matrix(self.loaded_property, item, select_slice=np.s_[:, start:stop],
                                       tensor=self.tensor_choice)

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

            return 0
        else:
            return 1

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
            return m*x + a

        # get the logarithmic dataset
        log_y = np.log10(data[0])
        log_x = np.log10(data[1])

        end_index = int(len(log_y) - 1)
        start_index = int(0.4*len(log_y))

        popt, pcov = curve_fit(func, log_x[start_index:end_index], log_y[start_index:end_index])  # fit linear regime

        if 0.85 < popt[0] < 1.15:
            self.loop_condition = True

        else:
            try:
                self.data_range = int(1.1*self.data_range)
                self.time = np.linspace(0.0, self.data_range * self.parent.time_step * self.parent.sample_rate,
                                        self.data_range)
                # end the calculation if the data range exceeds the relevant bounds
                if self.data_range > self.parent.number_of_configurations - self.correlation_time:
                    print("Trajectory not long enough to perform analysis.")
                    raise RangeExceeded
            except RangeExceeded:
                raise RangeExceeded

    def _fit_einstein_curve(self, data):
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

        def func(x, a):
            return x + a

        # get the logarithmic dataset
        log_y = np.log10(data[1][1:])
        log_x = np.log10(data[0][1:])

        min_end_index, max_end_index = int(0.7*len(log_y)), int(len(log_y) - 1)
        min_start_index, max_start_index = int(0.4*len(log_y)), int(0.6*len(log_y))

        for _ in range(100):
            end_index = random.randint(min_end_index, max_end_index)        # get a random end point
            start_index = random.randint(min_start_index, max_start_index)  # get a random start point

            popt, pcov = curve_fit(func, log_x[start_index:end_index], log_y[start_index:end_index])  # fit linear func
            fits.append(10**popt[0])

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

""" Parent class for different analysis """

import matplotlib.pyplot as plt

from mdsuite.utils.meta_functions import *


class Analysis:
    """ Parent class for analysis modules

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

    Methods
    -------
    _autocorrelation_time()
                        Perform autocorrelation time analysis on a data series to get the correct sampling rate.

    _collect_machine_properties

    _save_data(self, title, data)
                        Save a .npy file of generated data from the calculation.

    _plot_data(self)
                        Plot the generated data from the analysis and save in it.

    run_analysis(self)
                        Run the analysis being called. Will call all relevant methods.
    """

    def __init__(self, obj, plot=True, save=True, data_range=500, x_label=None, y_label=None, analysis_name=None):
        """ Python constructor """

        self.parent = obj  # Experiment object to get properties from
        self.data_range = data_range  # Data range over which to evaluate
        self.plot = plot  # Whether or not to plot the data and save a figure
        self.save = save  # Whether or not to save the calculated data (Default is true)
        self.loaded_property = None  # Which dataset to load
        self.batch_size = {}  # Size of the batch to use during the analysis
        self.n_batches = {}  # Number of batches to be calculated over
        self.machine_properties = None  # dictionary of machine properties to be evaluated at analysis run-time

        self.x_label = x_label  # x label of the figure
        self.y_label = y_label  # y label of the figure
        self.analysis_name = analysis_name  # what to save the figure as

    def _autocorrelation_time(self):
        """ get the autocorrelation time for the relevant property to ensure good error sampling """
        raise NotImplementedError  # Implemented in the child class

    def _collect_machine_properties(self):
        """ Collect properties of machine being used.

        This method will collect the properties of the machine being used and parse them to the relevant analysis in
        order to optimize the property computation.
        """

        self.machine_properties = get_machine_properties()  # load the machine properties
        memory_usage = []
        for item in self.parent.memory_requirements:
            memory_usage.append(self.parent.memory_requirements[item][self.loaded_property] /
                                self.parent.number_of_configurations)

        # Get the single frame memory usage in bytes
        serial_memory_usage = max(memory_usage) * self.data_range
        parallel_memory_usage = sum(memory_usage) * self.data_range

        # Update the batch_size attribute
        self.batch_size['Serial'] = int(np.floor(self.machine_properties['memory'] / serial_memory_usage))
        self.batch_size['Parallel'] = int(np.floor(self.machine_properties['memory'] / parallel_memory_usage))

        if self.batch_size['Serial']*self.data_range > self.parent.number_of_configurations:
            self.batch_size['Serial'] = int(np.floor(self.parent.number_of_configurations/self.data_range))
        if self.batch_size['Parallel']*self.data_range > self.parent.number_of_configurations:
            self.batch_size['Parallel'] = int(np.floor(self.parent.number_of_configurations/self.data_range))

        # Update the n_batches attribute
        self.n_batches['Serial'] = np.floor(((self.parent.number_of_configurations - self.data_range -
                                             1)/self.data_range) / self.batch_size['Serial']) + 1
        self.n_batches['Parallel'] = np.floor(((self.parent.number_of_configurations - self.data_range -
                                                   1)/self.data_range) / self.batch_size['Parallel']) + 1

    def _load_batch(self):
        """ Load a batch of data """
        raise NotImplementedError

    def _save_data(self, title, data):
        """ Save data to the save data directory """

        np.save(f"{self.parent.storage_path}/{self.parent.analysis_name}/data/{title}.npy", data)

    def _plot_data(self, title=None, manual=False):
        """ Plot the data generated during the analysis """

        if title is None:
            title = f"{self.analysis_name}"

        if manual:
            plt.savefig(f"{self.parent.storage_path}/{self.parent.analysis_name}/Figures/{title}.svg",
                        dpi=600, format='svg')
        else:
            plt.xlabel(rf'{self.x_label}')  # set the x label
            plt.ylabel(rf'{self.y_label}')  # set the y label
            plt.legend()  # enable the legend
            plt.savefig(f"{self.parent.storage_path}/{self.parent.analysis_name}/Figures/{title}.svg",
                        dpi=600, format='svg')

    def _perform_garbage_collection(self):
        """ Perform garbage collection after an analysis """
        raise NotImplementedError

    def run_analysis(self):
        """ Run the appropriate analysis


        Should follow the general outline detailed below:

        self._autocorrelation_time()  # Calculate the relevant autocorrelation time
        self._analysis()  # Can be diffusion coefficients or whatever is being calculated, but run the calculation
        self._error_anaysis  # Run an error analysis, could be done during the calculation, or may have to be for the
                                sake of memory.
        self._update_experiment  # Update the main experiment class with the calculated properties
        """
        raise NotImplementedError  # Implement in the child class

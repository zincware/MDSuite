"""
Parent class for different analysis

Summary
-------
"""

import abc
import random
import sys

import yaml
import h5py as hf
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import signal
from scipy.optimize import curve_fit
from tqdm import tqdm

from mdsuite.utils.exceptions import *
from mdsuite.utils.meta_functions import *
from mdsuite.plot_style.plot_style import apply_style
from mdsuite.memory_management.memory_manager import MemoryManager
from mdsuite.database.data_manager import DataManager
from mdsuite.database.database import Database

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdsuite.experiment.experiment import Experiment


class Calculator(metaclass=abc.ABCMeta):
    """
    Parent class for analysis modules

    Attributes
    ----------
    experiment : class object
            Class object of the experiment.
    plot : bool (default=True)
            Decision to plot the analysis.
    save : bool (default=True)
            Decision to save the generated tensor_values arrays.

    data_range : int (default=500)
            Range over which the property should be evaluated. This is not applicable to the current
            analysis as the full rdf will be calculated.
    x_label : str
            How to label the x axis of the saved plot.
    y_label : str
            How to label the y axis of the saved plot.
    analysis_name : str
            Name of the analysis. used in saving of the tensor_values and figure.
    batch_size : dict
            Size of batches to use in the analysis separated into parallel and serial components, i.e
            {'Serial': 100, 'Parallel': 50} for a two component, symmetric experiment.
    n_batches : dict
            Number of barthes to use as a dictionary for both serial and parallel implementations
    machine_properties : dict
            Devices available to MDSuite during analysis run. Has the following structure
            {'memory': x bytes, 'cpus': n_of_cores, 'gpus': name_of_gpu}

    """

    def __init__(self, experiment, plot=True, save=True, data_range=500, correlation_time=1):
        """

        Parameters
        ----------
        experiment
        plot
        save
        data_range
        x_label
        y_label
        analysis_name

        """
        self.experiment = experiment  # Experiment object to get properties from
        self.data_range = data_range  # Data range over which to evaluate
        self.plot = plot  # Whether or not to plot the tensor_values and save a figure
        self.save = save  # Whether or not to save the calculated tensor_values (Default is true)

        self.loaded_property = None  # Which dataset to load
        self.dependency = None

        self.batch_size: int  # Size of the batch to use during the analysis
        self.n_batches: int  # Number of batches to be calculated over
        self.remainder: int  # remainder after batching
        self.prefactor: float

        self.system_property = False  # is the calculation on a system property?
        self.multi_species = False  # does the calculation require loading of multiple species?
        self.species = None
        self.optimize = False
        self.batch_output_signature = None
        self.ensemble_output_signature = None

        self.machine_properties = None  # dictionary of machine properties to be evaluated at analysis run-time
        self.correlation_time = correlation_time  # correlation time of the property
        self.batch_loop = None  # Number of ensembles in each batch

        self.database = Database(name=os.path.join(self.experiment.database_path, "database.hdf5"))
        self.memory_manager: MemoryManager
        self.data_manager: DataManager

        self.x_label: str  # x label of the figure
        self.y_label: str  # y label of the figure
        self.analysis_name: str  # what to save the figure as

        self.database_group = None  # Which database_path group to save the tensor_values in
        self.time = np.linspace(0.0, self.data_range * self.experiment.time_step * self.experiment.sample_rate,
                                self.data_range)

        # Prevent $DISPLAY warnings on clusters.
        if self.experiment.cluster_mode:
            import matplotlib
            matplotlib.use('Agg')
        #else:
        #    apply_style()

    @abc.abstractmethod
    def _calculate_prefactor(self, species: str = None):
        """
        calculate the calculator pre-factor.

        Parameters
        ----------
        species : str
                Species property if required.
        Returns
        -------

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _apply_operation(self, data, index):
        """
        Perform operation on an ensemble.

        Parameters
        ----------
        One tensor_values range of tensor_values to operate on.

        Returns
        -------

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _apply_averaging_factor(self):
        """
        Apply an averaging factor to the tensor_values.
        Returns
        -------
        averaged copy of the tensor_values
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _post_operation_processes(self, species: str = None):
        """
        call the post-op processes
        Returns
        -------

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _update_output_signatures(self):
        """
        After having run _prepare managers, update the output signatures.

        Returns
        -------

        """
        raise NotImplementedError

    @staticmethod
    def _fit_einstein_curve(data: list):
        """
        Fit operation for Einstein calculations

        Parameters
        ----------
        data : list
                x and y tensor_values for the fitting [np.array, np.array] of (2, data_range)

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
                    x axis tensor_values for the fit
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

    @staticmethod
    def _convolution_op(data_a: tf.Tensor, data_v: tf.Tensor = None) -> tf.Tensor:
        """
        tf.numpy_function mapper of the np autocorrelation function

        Parameters
        ----------
        data_a : tf.Tensor
                Signal 1 of the correlation operation
        data_v : tf.Tensor
                Signal 2 of the correlation operation. If None, will be set to Signal 1 and autocorrelation is
                performed.

        Returns
        -------
        tf.Tensor
                Returns a tensor from the correlation operation.
        """
        if data_v is None:
            data_v = data_a

        def func(a, v):
            """
            Perform correlation on two tensor_values-sets.
            Parameters
            ----------
            a : np.ndarray
            v : np.ndarray

            Returns
            -------
            Returns the correlation of the two signals.
            """
            return sum([signal.correlate(a[:, idx], v[:, idx], mode="full", method='auto') for idx in range(3)])

        return tf.numpy_function(func=func, inp=[data_a, data_v], Tout=tf.float64)

    def _prepare_managers(self, data_path: list):
        """
        Prepare the memory and tensor_values monitors for calculation.

        Parameters
        ----------
        data_path : list
                List of tensor_values paths to load from the hdf5 database_path.

        Returns
        -------
        Updates the calculator class
        """
        self.memory_manager = MemoryManager(data_path=data_path,
                                            database=self.database,
                                            scaling_factor=1,
                                            memory_fraction=0.5)
        self.batch_size, self.n_batches, self.remainder = self.memory_manager.get_batch_size(system=self.system_property)
        self.ensemble_loop = self.memory_manager.get_ensemble_loop(self.data_range, self.correlation_time)
        self.data_manager = DataManager(data_path=data_path,
                                        database=self.database,
                                        data_range=self.data_range,
                                        batch_size=self.batch_size,
                                        n_batches=self.n_batches,
                                        ensemble_loop=self.ensemble_loop,
                                        correlation_time=self.correlation_time)
        self._update_output_signatures()

    def _save_data(self, title: str, data: np.array):
        """
        Save tensor_values to the save tensor_values directory

        Parameters
        ----------
        title : str
                name of the tensor_values to save. Usually this is just the analysis name. In the case of species specific
                analysis, this will be further appended to include the name of the species.
        data : np.array
                Data to be saved.
        """

        with hf.File(os.path.join(self.experiment.database_path, 'analysis_data.hdf5'), 'r+') as db:
            if title in db[self.database_group].keys():
                del db[self.database_group][title]
                db[self.database_group].create_dataset(title, data=data, dtype=float)
            else:
                db[self.database_group].create_dataset(title, data=data, dtype=float)

    def _plot_data(self, title: str = None, manual: bool = False):
        """
        Plot the tensor_values generated during the analysis
        """

        if title is None:
            title = f"{self.analysis_name}"

        if manual:
            plt.savefig(os.path.join(self.experiment.figures_path, f"{title}.svg"), dpi=600, format='svg')
        else:
            plt.xlabel(rf'{self.x_label}')  # set the x label
            plt.ylabel(rf'{self.y_label}')  # set the y label
            plt.legend()  # enable the legend
            plt.savefig(os.path.join(self.experiment.figures_path, f"{title}.svg"), dpi=600, format='svg')

    def _check_input(self):
        """
        Look for user input that would kill the analysis

        Returns
        -------
        status: int
            if 0, check failed, if 1, check passed.
        """
        if self.data_range > self.experiment.number_of_configurations - self.correlation_time:
            print("Data range is impossible for this experiment, reduce and try again")
            sys.exit(1)

    def _optimize_einstein_data_range(self, data: np.array):
        """
        Optimize the tensor_values range of a experiment using the Einstein method of calculation.

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
                    x axis tensor_values for the fit
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
                self.time = np.linspace(0.0, self.data_range * self.experiment.time_step * self.experiment.sample_rate,
                                        self.data_range)
                # end the calculation if the tensor_values range exceeds the relevant bounds
                if self.data_range > self.experiment.number_of_configurations - self.correlation_time:
                    print("Trajectory not long enough to perform analysis.")
                    raise RangeExceeded
            except RangeExceeded:
                raise RangeExceeded

    def _update_properties_file(self, item: str = None, sub_item: str = None, data: list = None, add: bool = False):
        """
        Update the experiment properties YAML file.
        """

        # Check if tensor_values has been given
        if data is None:
            print("No tensor_values provided")
            return

        with open(os.path.join(self.experiment.database_path, 'system_properties.yaml')) as pfr:
            properties = yaml.load(pfr, Loader=yaml.Loader)  # collect the tensor_values in the yaml file

        with open(os.path.join(self.experiment.database_path, 'system_properties.yaml'), 'w') as pfw:
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

    def _calculate_system_current(self):
        pass

    def _resolve_dependencies(self, dependency):
        """
        Resolve any calculation dependencies if possible.

        Parameters
        ----------
        dependency : str
                Name of the dependency to resolve.

        Returns
        -------

        """
        def _string_to_function(argument):
            """
            Select a transformation based on an input

            Parameters
            ----------
            argument : str
                    Name of the transformation required

            Returns
            -------
            transformation call.
            """
            switcher = {
                'Unwrapped_Positions': self._unwrap_choice(),
                'Translational_Dipole_Moment': 'TranslationalDipoleMoment',
                'Ionic_Current': 'IonicCurrent',
                'Integrated_Heat_Current': 'IntegratedHeatCurrent',
                'Thermal_Flux': 'ThermalFlux',
                'Momentum_Flux': 'MomentumFlux',
                'Kinaci_Integrated_Heat_Current': 'KinaciIntegratedHeatCurrent'
            }
            choice = switcher.get(argument, lambda: "Data not in database and can not be generated.")
            return choice

        transformation = _string_to_function(dependency)
        self.experiment.perform_transformation(transformation)

    def _unwrap_choice(self):
        """
        Unwrap either with indices or with box arrays.
        Returns
        -------

        """
        indices = self.database.check_existence('Box_Images')
        if indices:
            return 'UnwrapViaIndices'
        else:
            return 'UnwrapCoordinates'

    def _run_dependency_check(self):
        """
        Check to see if the necessary property exists and build it if required.

        Returns
        -------
        Will call transformations if required.
        """
        if self.dependency is not None:
            dependency = self.database.check_existence(self.dependency)
            if not dependency:
                self._resolve_dependencies(self.dependency)

        loaded_property = self.database.check_existence(self.loaded_property)
        if not loaded_property:
            self._resolve_dependencies(self.loaded_property)

    def perform_computation(self, data_path: list = None):
        """
        Perform the computation.
        Parameters
        ----------
        data_path : list
                if multi-species is present, the data_path is required to load the correct data.
        Returns
        -------
        Performs the analysis.
        """
        if self.system_property:
            self._calculate_prefactor()
            data_path = [join_path(self.loaded_property, self.loaded_property)]
            self._prepare_managers(data_path)
            batch_generator, batch_generator_args = self.data_manager.batch_generator()
            batch_data_set = tf.data.Dataset.from_generator(generator=batch_generator,
                                                            args=batch_generator_args,
                                                            output_signature=self.batch_output_signature)
            batch_data_set.prefetch(tf.data.experimental.AUTOTUNE)
            for batch_index, batch in tqdm(enumerate(batch_data_set), desc="Batch Loop", ncols=100):
                ensemble_generator, ensemble_generators_args = self.data_manager.ensemble_generator(
                    system=self.system_property)
                ensemble_data_set = tf.data.Dataset.from_generator(generator=ensemble_generator,
                                                                   args=ensemble_generators_args + (batch,),
                                                                   output_signature=self.ensemble_output_signature)
                for ensemble_index, ensemble in tqdm(enumerate(ensemble_data_set), desc="Ensemble Loop", ncols=100):
                    self._apply_operation(ensemble, ensemble_index)

            self._apply_averaging_factor()
            self._post_operation_processes()

        elif self.multi_species:
            self._calculate_prefactor()
            self._prepare_managers(data_path)
            batch_generator, batch_generator_args = self.data_manager.batch_generator()
            batch_data_set = tf.data.Dataset.from_generator(generator=batch_generator,
                                                            args=batch_generator_args,
                                                            output_signature=self.batch_output_signature)
            batch_data_set.prefetch(tf.data.experimental.AUTOTUNE)
            for batch_index, batch in tqdm(enumerate(batch_data_set), desc="Batch Loop", ncols=100):
                self._apply_operation(batch, batch_index)

        else:
            for species in self.species:
                self._calculate_prefactor(species)
                data_path = [join_path(species, self.loaded_property)]
                self._prepare_managers(data_path)
                batch_generator, batch_generator_args = self.data_manager.batch_generator()
                batch_data_set = tf.data.Dataset.from_generator(generator=batch_generator,
                                                                args=batch_generator_args,
                                                                output_signature=self.batch_output_signature)
                batch_data_set.prefetch(tf.data.experimental.AUTOTUNE)
                for batch_index, batch in tqdm(enumerate(batch_data_set), desc="Batch Loop", ncols=100):
                    ensemble_generator, ensemble_generators_args = self.data_manager.ensemble_generator()
                    ensemble_data_set = tf.data.Dataset.from_generator(generator=ensemble_generator,
                                                                       args=ensemble_generators_args + (batch,),
                                                                       output_signature=self.ensemble_output_signature)
                    for ensemble_index, ensemble in tqdm(enumerate(ensemble_data_set), desc="Ensemble Loop"):
                        self._apply_operation(ensemble, ensemble_index)

                self._apply_averaging_factor()
                self._post_operation_processes(species)

    def run_analysis(self):
        """
        Run the appropriate analysis
        Should follow the general outline detailed below:
        self._analysis()  # Can be diffusion coefficients or whatever is being calculated, but run the calculation
        self._error_analysis  # Run an error analysis, could be done during the calculation.
        self._update_experiment  # Update the main experiment class with the calculated properties

        """
        self._check_input()
        self._run_dependency_check()

        if self.optimize:
            pass
        else:
            self.perform_computation()

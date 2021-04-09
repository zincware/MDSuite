"""
Parent class for different analysis

Summary
-------
"""

import abc
import random
from typing import TYPE_CHECKING
import sys

import h5py as hf
import matplotlib.figure
import matplotlib.pyplot as plt

from matplotlib.axes._subplots import Axes

import tensorflow as tf
import yaml

from mdsuite.plot_style.plot_style import apply_style
from mdsuite.utils.exceptions import *
from mdsuite.utils.meta_functions import *
from scipy import signal
from scipy.optimize import curve_fit
from tqdm import tqdm

if TYPE_CHECKING:
    from mdsuite.experiment.experiment import Experiment
from mdsuite.utils.exceptions import *
from mdsuite.utils.meta_functions import *
from mdsuite.plot_style.plot_style import apply_style  # TODO killed the code.
from mdsuite.memory_management.memory_manager import MemoryManager
from mdsuite.database.data_manager import DataManager
from mdsuite.database.database import Database
from mdsuite.calculators.computations_dict import switcher_transformations


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
    batch_size : dict
            Size of batches to use in the analysis separated into parallel and serial components, i.e
            {'Serial': 100, 'Parallel': 50} for a two component, symmetric experiment.
    n_batches : dict
            Number of barthes to use as a dictionary for both serial and parallel implementations
    """

    def __init__(self, experiment, plot=True, save=True, data_range=500, correlation_time=1, atom_selection=np.s_[:]):
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

        self.atom_selection = atom_selection

        self.loaded_property = None  # Which dataset to load
        self.dependency = None

        self.batch_size: int  # Size of the batch to use during the analysis
        self.n_batches: int  # Number of batches to be calculated over
        self.remainder: int  # remainder after batching
        self.prefactor: float

        self.system_property = False  # is the calculation on a system property?
        self.multi_species = False  # does the calculation require loading of multiple species?
        self.experimental = False  # experimental calculator.
        self.species = None
        self.optimize = False
        self.batch_output_signature = None
        self.ensemble_output_signature = None
        self.correlation_time = correlation_time  # correlation time of the property

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

        return [np.mean(fits), np.std(fits)]

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
                                            memory_fraction=0.5,
                                            scale_function=self.scale_function)
        self.batch_size, self.n_batches, self.remainder = self.memory_manager.get_batch_size(system=self.system_property)

        self.ensemble_loop = self.memory_manager.get_ensemble_loop(self.data_range, self.correlation_time)
        self.data_manager = DataManager(data_path=data_path,
                                        database=self.database,
                                        data_range=self.data_range,
                                        batch_size=self.batch_size,
                                        n_batches=self.n_batches,
                                        ensemble_loop=self.ensemble_loop,
                                        correlation_time=self.correlation_time,
                                        remainder=self.remainder,
                                        atom_selection=self.atom_selection
                                        )
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

    def _plot_fig(self, fig: matplotlib.figure.Figure, ax: Axes, title: str = None, dpi: int = 600,
                  filetype: str = 'svg'):
        """Class based plotting using fig, ax = plt.subplots

        Parameters
        ----------
        fig: matplotlib figure
        ax: matplotlib subplot axes
            currently only a single axes is supported. Subplots aren't yet!
        title: str
            Name of the plot
        dpi: int
            matplotlib dpi resolution
        filetype: str
            matplotlib filetype / format
        """

        if title is None:
            title = f"{self.analysis_name}"

        ax.set_xlabel(rf'{self.x_label}')
        ax.set_ylabel(rf'{self.y_label}')
        ax.legend()
        fig.set_facecolor("w")
        fig.show()

        fig.savefig(os.path.join(self.experiment.figures_path, f"{title}.svg"), dpi=dpi, format=filetype)

    def _plot_data(self, title: str = None, manual: bool = False, dpi: int = 600):
        """
        Plot the tensor_values generated during the analysis
        """

        if title is None:
            title = f"{self.analysis_name}"

        if manual:
            plt.savefig(os.path.join(self.experiment.figures_path, f"{title}.svg"), dpi=dpi, format='svg')
        else:
            plt.xlabel(rf'{self.x_label}')  # set the x label
            plt.ylabel(rf'{self.y_label}')  # set the y label
            plt.legend()  # enable the legend
            plt.savefig(os.path.join(self.experiment.figures_path, f"{title}.svg"), dpi=dpi, format='svg')

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

        results = self.experiment.results

        # TODO: improve this if else blocks. I am sure it can be done in a more elegant way
        if item is None:
            results[self.database_group][self.analysis_name] = str(data)
        elif sub_item is None:
            results[self.database_group][self.analysis_name][item] = str(data)
        else:
            if add:
                results[self.database_group][self.analysis_name][item] = {}
                results[self.database_group][self.analysis_name][item][sub_item] = str(data)

        self.experiment.results = results

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

            switcher_unwrapping = {'Unwrapped_Positions': self._unwrap_choice(), }

            switcher = {**switcher_unwrapping, **switcher_transformations}

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

    def perform_computation(self):
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
            batch_generator, batch_generator_args = self.data_manager.batch_generator(system=self.system_property)
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

        elif self.experimental:
            data_path = [join_path(species, self.loaded_property) for species in self.experiment.species]
            self._prepare_managers(data_path)
            self.run_experimental_analysis()

        else:
            for species in self.species:
                self._calculate_prefactor(species)
                data_path = [join_path(species, self.loaded_property)]
                self._prepare_managers(data_path)
                batch_generator, batch_generator_args = self.data_manager.batch_generator()
                batch_data_set = tf.data.Dataset.from_generator(generator=batch_generator,
                                                                args=batch_generator_args,
                                                                output_signature=self.batch_output_signature)
                batch_data_set = batch_data_set.prefetch(tf.data.experimental.AUTOTUNE)
                for batch_index, batch in tqdm(enumerate(batch_data_set), desc="Batch Loop",
                                               ncols=100, total=self.n_batches):
                    ensemble_generator, ensemble_generators_args = self.data_manager.ensemble_generator()
                    ensemble_data_set = tf.data.Dataset.from_generator(generator=ensemble_generator,
                                                                       args=ensemble_generators_args + (batch,),
                                                                       output_signature=self.ensemble_output_signature)
                    ensemble_data_set = ensemble_data_set.prefetch(tf.data.experimental.AUTOTUNE)
                    for ensemble_index, ensemble in enumerate(ensemble_data_set):
                        self._apply_operation(ensemble, ensemble_index)

                self._apply_averaging_factor()
                self._post_operation_processes(species)

    def run_analysis(self):
        """
        Run the appropriate analysis
        """
        self._check_input()
        self._run_dependency_check()

        if self.optimize:
            pass
        else:
            self.perform_computation()

    def run_experimental_analysis(self):
        """
        For experimental methods
        Returns
        -------

        """
        raise NotImplementedError

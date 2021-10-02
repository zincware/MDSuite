"""
MDSuite: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
"""
from __future__ import annotations

import logging
import numpy as np
import random
from pathlib import Path
import tensorflow as tf
from scipy.optimize import curve_fit
from mdsuite.visualizer.d2_data_visualization import DataVisualizer2D
from mdsuite.utils.exceptions import RangeExceeded
from mdsuite.utils.meta_functions import join_path
from mdsuite.memory_management.memory_manager import MemoryManager
from mdsuite.database.data_manager import DataManager
from mdsuite.database.simulation_database import Database
from mdsuite.calculators.transformations_reference import switcher_transformations
from mdsuite.database.calculator_database import CalculatorDatabase
import mdsuite.database.scheme as db
from tqdm import tqdm
from typing import Union, List, Dict
import warnings

import functools

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdsuite import Experiment

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


def call(func):
    """Decorator for the calculator call method

    This decorator provides a unified approach for handling run_computation and
    load_data for a single or multiple experiments.
    It handles the `run.<calc>()` method, iterates over experiments and
    loads data if requested! Therefore, the __call__ method does not and can
    not return any values anymore!


    Notes
    -----
    When calling the calculator it will check if a computation with the given
    user arguments was already performed:
    >>> Calculator.get_computation_data() is not None

    if no computations are available it will
    1. prepare a database entry
    >>> Calculator.prepare_db_entry()
    2. save the user arguments
    >>> Calculator.save_computation_args()
    3. Run the analysis
    >>> Calculator.run_analysis()
    4. Save all the data to the database
    >>> Calculator.save_db_data()
    5. Finally query the the data from the database and pass them to the user / plotting
    >>> data = Calculator.get_computation_data()




    Parameters
    ----------
    func: Calculator.__call__ method

    Returns
    -------
    decorated __call__ method

    """

    @functools.wraps(func)
    def inner(
        self, *args, **kwargs
    ) -> Union[db.Computation, Dict[str, db.Computation]]:
        """Manage the call method

        Parameters
        ----------
        self: Calculator

        Returns
        -------
        data:
            A dictionary of shape {name: data} when called from the project class
            A list of [data] when called directly from the experiment class
        """
        # This is only true, when called via project.experiments.<exp>.run,
        #  otherwise the experiment will be None
        return_dict = self.experiment is None

        out = {}
        for experiment in self.experiments:
            self.experiment = experiment
            self.clean_cache()
            # pass the user args to the calculator
            func(self, *args, **kwargs)
            data = self.get_computation_data()
            if data is None:
                # new calculation will be performed
                self.prepare_db_entry()
                self.save_computation_args()
                self.run_analysis()
                self.save_db_data()
                # Need to reset the user args, if they got change
                # or set to defaults, e.g. n_configurations = - 1 so
                # that they match the query
                func(self, *args, **kwargs)
                data = self.get_computation_data()

            if self.plot:
                """Plot the data"""
                self.plotter = DataVisualizer2D(title=self.analysis_name)
                self.plot_data(data.data_dict)
                self.plotter.grid_show(self.plot_array)

            out[self.experiment.name] = data

        if return_dict:
            return out
        else:
            return out[self.experiment.name]

    return inner


class Calculator(CalculatorDatabase):
    """
    Parent class for analysis modules

    Attributes
    ----------
    experiment : Experiment
            Class object of the experiment.
    plot : bool (default=True)
            Decision to plot the analysis.
    save : bool (default=True)
            Decision to save the generated tensor_values arrays.
    batch_size : dict
            Size of batches to use in the analysis separated into parallel and
             serial components, i.e {'Serial': 100, 'Parallel': 50} for a two
             component, symmetric experiment.
    n_batches : dict
            Number of barthes to use as a dictionary for both serial and
            parallel implementations
    plot_array : list
            A list of plot objects to be show together at the end of the
            species loop.
    """

    def __init__(
        self,
        experiment: Experiment = None,
        experiments: List[Experiment] = None,
        plot: bool = True,
        save: bool = True,
        atom_selection: object = np.s_[:],
        gpu: bool = False,
        load_data: bool = False,
    ):
        """
        Constructor for the calculator class.

        Parameters
        ----------
        experiment : Experiment
                Experiment object to update.
        experiments: List[Experiment]:
                List of experiments, for that the calculation should be executed
        plot : bool
                If true, analysis is plotted.
        save : bool
                If true, the analysis is saved.
        atom_selection : np.s_
                Atoms to perform the analysis on.
        gpu : bool
                If true, reduce memory usage to what is allowed on the system
                GPU.
        """
        # Set upon instantiation of parent class
        super().__init__(experiment)
        self.experiment: Experiment = experiment
        self.experiments: List[Experiment] = experiments
        self.trial_pp = False
        # Setting the experiment value supersedes setting experiments
        if self.experiment is not None:
            self.experiments = [self.experiment]

        self.plot = plot
        self.save = save
        self.atom_selection = atom_selection
        self.gpu = gpu
        self.load_data = load_data

        # Set to default by class, over-written in child classes or during operations
        self.system_property = False
        self.multi_species = False
        self.post_generation = False
        self.experimental = False
        self.optimize = False
        self.loaded_property = None
        self.dependency = None
        self.scale_function = None
        self.batch_output_signature = None
        self.ensemble_output_signature = None
        # self.species = None
        # all species that are used in the calculation
        self.selected_species = None
        # the selected species which the current calculation iteration is performed on
        self.database_group = None
        self.analysis_name = None
        self.tau_values = None
        self.time = None
        self.data_resolution = None
        self.plotter = None
        # e.g. [diffusion_coefficient, uncertainty]
        self.result_keys = None
        # e.g., [time, msd]
        self.result_series_keys = None

        # Set during operation or by child class
        self.batch_size: int
        self.n_batches: int
        self.remainder: int
        self.prefactor: float
        self.memory_manager: MemoryManager
        self.data_manager: DataManager
        self.x_label: str
        self.y_label: str
        self.analysis_name: str
        self.minibatch: bool

        self.x_label: str
        self.y_label: str
        self.analysis_name: str
        self.plot_array = []

        self.database_group = None
        self.analysis_name = None

        # Properties
        self._dtype = tf.float64
        self._database = None

    @property
    def database(self):
        """Get the database based on the experiment database path"""
        if self._database is None:
            self._database = Database(
                name=Path(self.experiment.database_path, "database.hdf5").as_posix()
            )
        return self._database

    # def update_user_args(
    #     self,
    #     plot: bool,
    #     data_range: int = 500,
    #     correlation_time: int = 1,
    #     atom_selection: object = np.s_[:],
    #     tau_values: Union[int, List, Any] = np.s_[:],
    #     gpu: bool = False,
    #     *args,
    #     **kwargs,
    # ):
    #     """
    #     Update the user args that are given by the __call__ method of the
    #     calculator.
    #
    #     Parameters
    #     ----------
    #     plot : bool
    #             If true, analysis is plotted.
    #     save : bool
    #             If true, the analysis is saved.
    #     data_range : int
    #             Data range over which to compute.
    #     correlation_time : int
    #             Correlation time to use in the analysis.
    #     atom_selection : object
    #             Atoms to perform the analysis on.
    #     gpu : bool
    #             If true, reduce memory usage to what is allowed on the system
    #             GPU.
    #     """
    #
    #     # Prevent $DISPLAY warnings on clusters.
    #     if self.experiment.cluster_mode:
    #         import matplotlib
    #
    #         matplotlib.use("Agg")
    #
    #     self.data_range = data_range
    #     self.plot = plot
    #     self.gpu = gpu
    #     self.tau_values = tau_values
    #     self.correlation_time = correlation_time
    #     self.atom_selection = atom_selection
    #
    #     # attributes based on user args
    #     self.time = self._handle_tau_values()  # process selected tau values.

    def _calculate_prefactor(self, species: Union[str, tuple] = None):
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

    def _apply_averaging_factor(self):
        """
        Apply an averaging factor to the tensor_values.
        Returns
        -------
        averaged copy of the tensor_values
        """
        raise NotImplementedError

    def _post_operation_processes(self, species: Union[str, tuple] = None):
        """
        call the post-op processes

        Parameters
        ----------
        species : Union[str, tuple]
                List or tuple of species fo which this post-operation process
                is being applied.
        Returns
        -------

        """
        raise NotImplementedError

    def _update_output_signatures(self):
        """
        After having run _prepare managers, update the output signatures.

        Returns
        -------

        """
        pass

    @staticmethod
    def _fit_einstein_curve(data: list) -> list:
        """
        Fit operation for Einstein calculations

        Parameters
        ----------
        data : list
                x and y tensor_values for the fitting [np.array, np.array] of
                (2, data_range)

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
                    scalar offset, also the y-intercept for those who did not
                    get much maths in school.

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
            end_index = random.randint(min_end_index, max_end_index)
            start_index = random.randint(min_start_index, max_start_index)

            popt, pcov = curve_fit(
                func, log_x[start_index:end_index], log_y[start_index:end_index]
            )
            fits.append(10 ** popt[1])

        return [np.mean(fits), np.std(fits)]

    @staticmethod
    def _update_species_type_dict(dictionary: dict, path_list: list, dimension: int):
        """
        Update a type spec dictionary for a species input.

        Parameters
        ----------
        dictionary : dict
                Dictionary to append
        path_list : list
                List of paths for the dictionary
        dimension : int
                Dimension of the property
        Returns
        -------
        type dict : dict
                Dictionary for the type spec.
        """
        for item in path_list:
            dictionary[str.encode(item)] = tf.TensorSpec(
                shape=(None, None, dimension), dtype=tf.float64
            )

        return dictionary

    def _handle_tau_values(self) -> np.array:
        """
        Handle the parsing of custom tau values.

        Returns
        -------
        times : np.array
            The time values corresponding to the selected tau values
        """
        if isinstance(self.args.tau_values, int):
            self.data_resolution = self.args.tau_values
            self.args.tau_values = np.linspace(
                0, self.args.data_range - 1, self.args.tau_values, dtype=int
            )
        if isinstance(self.args.tau_values, list) or isinstance(
            self.args.tau_values, np.ndarray
        ):
            self.data_resolution = len(self.args.tau_values)
            self.args.data_range = self.args.tau_values[-1] + 1
        if isinstance(self.args.tau_values, slice):
            self.args.tau_values = np.linspace(
                0, self.args.data_range - 1, self.args.data_range, dtype=int
            )[self.args.tau_values]
            self.data_resolution = len(self.args.tau_values)

        times = (
            np.asarray(self.args.tau_values)
            * self.experiment.time_step
            * self.experiment.sample_rate
        )

        return times

    def _prepare_managers(self, data_path: list, correct: bool = False):
        """
        Prepare the memory and tensor_values monitors for calculation.

        Parameters
        ----------
        data_path : list
                List of tensor_values paths to load from the hdf5
                database_path.

        Returns
        -------
        Updates the calculator class
        """
        self.memory_manager = MemoryManager(
            data_path=data_path,
            database=self.database,
            memory_fraction=0.8,
            scale_function=self.scale_function,
            gpu=self.gpu,
        )
        (
            self.batch_size,
            self.n_batches,
            self.remainder,
        ) = self.memory_manager.get_batch_size(system=self.system_property)

        self.ensemble_loop, minibatch = self.memory_manager.get_ensemble_loop(
            self.args.data_range, self.args.correlation_time
        )
        if minibatch:
            self.batch_size = self.memory_manager.batch_size
            self.n_batches = self.memory_manager.n_batches
            self.remainder = self.memory_manager.remainder

        if correct:
            self._correct_batch_properties()

        self.data_manager = DataManager(
            data_path=data_path,
            database=self.database,
            data_range=self.args.data_range,
            batch_size=self.batch_size,
            n_batches=self.n_batches,
            ensemble_loop=self.ensemble_loop,
            correlation_time=self.args.correlation_time,
            remainder=self.remainder,
            atom_selection=self.args.atom_selection,
            minibatch=minibatch,
            atom_batch_size=self.memory_manager.atom_batch_size,
            n_atom_batches=self.memory_manager.n_atom_batches,
            atom_remainder=self.memory_manager.atom_remainder,
        )

    def _build_table_name(self, species: str = "System"):
        """
        Build the sql table name for the data storage.

        Parameters
        ----------
        species : str
                In the case of a species specific analysis, make sure a species
                is put here. Otherwise, it is set to System.
        Returns
        -------
        name : str
                A correctly formatted name for the SQL database.
        """
        return f"{self.database_group}_{self.analysis_name}_{self.args.data_range}_{species}"

    def run_visualization(self, x_data, y_data, title: str, layouts: object = None):
        """
        Run a visualization session on the data.

        Parameters
        ----------
        x_data
        y_data
        title
        span

        Returns
        -------

        """
        self.plot_array.append(
            self.plotter.construct_plot(
                x_data=x_data,
                y_data=y_data,
                title=title,
                x_label=self.x_label,
                y_label=self.y_label,
                layouts=layouts,
            )
        )

    def check_input(self):
        """
        Look for user input that would kill the analysis

        Returns
        -------
        status: int
            if 0, check failed, if 1, check passed.
        """

        raise NotImplementedError("Please implement check input in child class")

    def _optimize_einstein_data_range(self, data: np.array):
        """
        Optimize the tensor_values range of a experiment using the Einstein
        method of calculation.

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
                    scalar offset, also the y-intercept for those who did not
                    get much maths in school.

            Returns
            -------

            """

            return m * x + a

        # get the logarithmic dataset
        log_y = np.log10(data[0])
        log_x = np.log10(data[1])

        end_index = int(len(log_y) - 1)
        start_index = int(0.4 * len(log_y))

        popt, pcov = curve_fit(
            func, log_x[start_index:end_index], log_y[start_index:end_index]
        )

        if 0.85 < popt[0] < 1.15:
            self.loop_condition = True

        else:
            try:
                self.args.data_range = int(1.1 * self.args.data_range)
                self.time = np.linspace(
                    0.0,
                    self.args.data_range
                    * self.experiment.time_step
                    * self.experiment.sample_rate,
                    self.args.data_range,
                )

                # end the calculation if the tensor_values range exceeds the relevant
                # bounds
                if (
                    self.args.data_range
                    > self.experiment.number_of_configurations
                    - self.args.correlation_time
                ):
                    print("Trajectory not long enough to perform analysis.")
                    raise RangeExceeded
            except RangeExceeded:
                raise RangeExceeded

    def _update_properties_file(self, parameters: dict, delete_duplicate: bool = True):
        """
        Update the experiment properties YAML file.
        """
        log.warning(
            "Using depreciated method `_update_properties_file` \t Please use"
            " `update_database` instead."
        )
        self.update_database(parameters=parameters, delete_duplicate=delete_duplicate)

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

            switcher_unwrapping = {
                "Unwrapped_Positions": self._unwrap_choice(),
            }

            # add the other transformations and merge the dictionaries
            switcher = {**switcher_unwrapping, **switcher_transformations}

            choice = switcher.get(
                argument, lambda: "Data not in database and can not be generated."
            )
            return choice

        transformation = _string_to_function(dependency)
        self.experiment.perform_transformation(transformation)

    def _unwrap_choice(self):
        """
        Unwrap either with indices or with box arrays.
        Returns
        -------

        """
        indices = self.database.check_existence("Box_Images")
        if indices:
            return "UnwrapViaIndices"
        else:
            return "UnwrapCoordinates"

    def _run_dependency_check(self):
        """
        Check to see if the necessary property exists and build it if required.

        Returns
        -------
        Will call transformations if required.
        """

        if self.loaded_property is None:
            return

        if self.dependency is not None:
            dependency = self.database.check_existence(self.dependency[0])
            if not dependency:
                self._resolve_dependencies(self.dependency[0])

        loaded_property = self.database.check_existence(self.loaded_property[0])
        if not loaded_property:
            self._resolve_dependencies(self.loaded_property[0])

    def perform_computation(self):
        """
        Perform the computation.
        Returns
        -------
        Performs the analysis.
        """
        if self.post_generation:
            self.run_post_generation_analysis()
        else:
            self.run_calculator()

    def run_post_generation_analysis(self):
        """
        Run a post-generation analysis.
        """
        raise NotImplementedError

    def run_calculator(self):
        """
        Run a standard calculation, i.e, not one that involves loading data from the
        Project SQL database.

        Returns
        -------

        """
        raise NotImplementedError

    def run_analysis(self):
        """
        Run the appropriate analysis
        """
        self._run_dependency_check()
        if self.experimental:
            log.warning(
                "This is an experimental calculator. Please see the "
                "documentation before using the results."
            )
        self.perform_computation()

    @property
    def dtype(self):
        """Get the dtype used for the calculator"""
        return self._dtype

    def plot_data(self, data):
        """
        Plot the data coming from the database

        Parameters
        ----------
        data: db.Compution.data_dict
                associated with the current project
        """
        for selected_species, val in data.items():
            self.run_visualization(
                x_data=np.array(val[self.result_series_keys[0]])
                * self.experiment.units["time"],
                y_data=np.array(val[self.result_series_keys[1]])
                * self.experiment.units["time"],
                title=(
                    f"{selected_species}: {val[self.result_keys[0]]: 0.3E} +-"
                    f" {val[self.result_keys[1]]: 0.3E}"
                ),
            )

    def _correct_batch_properties(self):
        """
        Fix batch properties.
        """
        raise NotImplementedError

    def get_batch_dataset(
        self,
        subject_list: list = None,
        loop_array: np.ndarray = None,
        correct: bool = False,
    ):
        """
        Collect the batch loop dataset

        Parameters
        ----------
        subject_list : list (default = None)
                A str of subjects to collect data for in case this is necessary. The
                method will first try to split this string by an '_' in the case where
                tuples have been parsed. If None, the method assumes that this is a
                system calculator and returns a generator appropriate to such an
                analysis.
                e.g. subject = ['Na']
                     subject = ['Na', 'Cl', 'K']
                     subject = ['Ionic_Current']
        loop_array : np.ndarray (default = None)
                If this is not None, elements of this array will be looped over in
                in the batches which load data at their indices. For example,
                    loop_array = [[1, 4, 7], [10, 13, 16], [19, 21, 24]]
                In this case, in the fist batch, configurations 1, 4, and 7 will be
                loaded for the analysis. This is particularly important in the
                structural properties.

        Returns
        -------
        dataset : tf.data.Dataset
                A TensorFlow dataset for the batch loop to be iterated over.

        """
        path_list = [join_path(item, self.loaded_property[0]) for item in subject_list]
        self._prepare_managers(path_list, correct=correct)

        type_spec = {}
        for item in subject_list:
            dict_ref = "/".join([item, self.loaded_property[0]])
            type_spec[str.encode(dict_ref)] = tf.TensorSpec(
                shape=self.loaded_property[1], dtype=self.dtype
            )
        type_spec[str.encode("data_size")] = tf.TensorSpec(shape=(), dtype=tf.int32)

        batch_generator, batch_generator_args = self.data_manager.batch_generator(
            dictionary=True, system=self.system_property, loop_array=loop_array
        )
        ds = tf.data.Dataset.from_generator(
            generator=batch_generator,
            args=batch_generator_args,
            output_signature=type_spec,
        )

        return ds.prefetch(tf.data.AUTOTUNE)

    def get_ensemble_dataset(self, batch: dict, subject: str, split: bool = False):
        """
        Collect the ensemble loop dataset.

        Parameters
        ----------
        batch : tf.Tensor
                A batch of data to be looped over in ensembles.

        Returns
        -------
        dataset : tf.data.Dataset
                A TensorFlow dataset object for the ensemble loop to be iterated over.

        """
        (
            ensemble_generator,
            ensemble_generators_args,
        ) = self.data_manager.ensemble_generator(
            dictionary=True, glob_data=batch, system=self.system_property
        )

        type_spec = {}

        if split:
            loop_list = subject.split("_")
        else:
            loop_list = [subject]
        for item in loop_list:
            dict_ref = "/".join([item, self.loaded_property[0]])
            type_spec[str.encode(dict_ref)] = tf.TensorSpec(
                shape=self.loaded_property[1], dtype=self.dtype
            )

        ds = tf.data.Dataset.from_generator(
            generator=ensemble_generator,
            args=ensemble_generators_args,
            output_signature=type_spec,
        )

        return ds.prefetch(tf.data.AUTOTUNE)

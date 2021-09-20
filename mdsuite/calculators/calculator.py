"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Parent class for different analysis

Summary
-------
"""
from __future__ import annotations

import logging
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import pandas as pd
from scipy.optimize import curve_fit
from mdsuite.visualizer.d2_data_visualization import DataVisualizer2D
from mdsuite.utils.exceptions import RangeExceeded
from mdsuite.utils.meta_functions import join_path
from mdsuite.memory_management.memory_manager import MemoryManager
from mdsuite.database.data_manager import DataManager
from mdsuite.database.simulation_database import Database
from mdsuite.calculators.transformations_reference import \
    switcher_transformations
from mdsuite.database.calculator_database import CalculatorDatabase
from tqdm import tqdm
from typing import Union, List, Any

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdsuite import Experiment

log = logging.getLogger(__name__)


def call(func):
    """Decorator for the calculator call method

    This decorator provides a unified approach for handling run_computation and load_data for a single or multiple
    experiments.
    It handles the `run_computation()` method, iterates over experiments and loads data if requested!
    Therefore, the __call__ method does not and can not return any values anymore!

    Parameters
    ----------
    func: Calculator.__call__ method

    Returns
    -------
    decorated __call__ method

    """

    def inner(self, *args, **kwargs):
        """Manage the call method

        Parameters
        ----------
        self: Calculator


        Returns
        -------
        data:
            A dictionary of shape {name: data} for multiple len(experiments) > 1 or otherwise just data
        """
        out = {}
        for experiment in self.experiments:
            self.experiment = experiment
            func(self, *args, **kwargs)
            if self.load_data:
                out[self.experiment.name] = self.experiment.export_property_data(
                    {"Analysis": self.analysis_name, "experiment": self.experiment.name}
                )
            else:
                out[self.experiment.name] = self.run_analysis()

        if len(self.experiments) > 1:
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

    data_range : int (default=500)
            Range over which the property should be evaluated. This is not
            applicable to the current analysis as the full rdf will be
            calculated.
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

    def __init__(self, experiment: Experiment = None, experiments: List[Experiment] = None, plot: bool = True,
                 save: bool = True, data_range: int = 500,
                 correlation_time: int = 1, atom_selection: object = np.s_[:], export: bool = True, gpu: bool = False,
                 load_data: bool = False):
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
        data_range : int
                Data range over which to compute.
        correlation_time : int
                Correlation time to use in the analysis.
        atom_selection : np.s_
                Atoms to perform the analysis on.
        export : bool
                If true, analysis results are exported to a csv file.
        gpu : bool
                If true, reduce memory usage to what is allowed on the system
                GPU.
        """
        # Set upon instantiation of parent class
        super().__init__(experiment)
        self.experiment: Experiment = experiment
        self.experiments: List[Experiment] = experiments

        # Setting the experiment value supersedes setting experiments
        if self.experiment is not None:
            self.experiments = [self.experiment]

        self.data_range = data_range
        self.plot = plot
        self.save = save
        self.export = export
        self.atom_selection = atom_selection
        self.correlation_time = correlation_time
        self.database = None
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
        self.species = None
        self.database_group = None
        self.analysis_name = None
        self.tau_values = None
        self.time = None
        self.data_resolution = None

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
        self.last_iteration = False
        self.plotter = DataVisualizer2D(title=self.analysis_name)

        self.database_group = None
        self.analysis_name = None

        # Properties
        self._dtype = tf.float64

    def update_user_args(self,
                         plot: bool = True,
                         save: bool = True,
                         data_range: int = 500,
                         correlation_time: int = 1,
                         atom_selection: object = np.s_[:],
                         tau_values: Union[int, List, Any] = np.s_[:],
                         export: bool = True,
                         gpu: bool = False,
                         *args,
                         **kwargs):
        """
        Update the user args that are given by the __call__ method of the
        calculator.

        Parameters
        ----------
        plot : bool
                If true, analysis is plotted.
        save : bool
                If true, the analysis is saved.
        data_range : int
                Data range over which to compute.
        correlation_time : int
                Correlation time to use in the analysis.
        atom_selection : object
                Atoms to perform the analysis on.
        export : bool
                If true, analysis results are exported to a csv file.
        gpu : bool
                If true, reduce memory usage to what is allowed on the system
                GPU.
        """
        # everything related to self.experiment can not be in the __init__ because there can be multiple experiments
        self.database = Database(
            name=Path(self.experiment.database_path, "database.hdf5").as_posix())

        # Prevent $DISPLAY warnings on clusters.
        if self.experiment.cluster_mode:
            import matplotlib
            matplotlib.use('Agg')

        self.data_range = data_range
        self.plot = plot
        self.save = save
        self.export = export
        self.gpu = gpu
        self.tau_values = tau_values
        self.correlation_time = correlation_time
        self.atom_selection = atom_selection

        # attributes based on user args
        self.time = self._handle_tau_values()  # process selected tau values.

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
        raise NotImplementedError

    @staticmethod
    def _fit_einstein_curve(data: list):
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

        min_end_index, max_end_index = int(0.8 * len(log_y)), \
                                       int(len(log_y) - 1)
        min_start_index, max_start_index = int(0.3 * len(log_y)), \
                                           int(0.5 * len(log_y))

        for _ in range(100):
            end_index = random.randint(min_end_index, max_end_index)
            start_index = random.randint(min_start_index, max_start_index)

            popt, pcov = curve_fit(func, log_x[start_index:end_index],
                                   log_y[start_index:end_index])
            fits.append(10 ** popt[1])

        return [np.mean(fits), np.std(fits)]

    def _update_species_type_dict(dictionary: dict,
                                  path_list: list,
                                  dimension: int):
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
            dictionary[str.encode(item)] = tf.TensorSpec(shape=(None, None,
                                                                dimension),
                                                         dtype=tf.float64)

        return dictionary

    def _build_pandas_dataframe(x: np.array, y: np.array) -> pd.DataFrame:
        """
        Build a pandas dataframe with x and y data.

        Parameters
        ----------
        x : np.array
                x data to go into the data frame
        y : np.array
                y data to go into the data frame

        Returns
        -------
        data : pd.DataFrame
                Pandas data frame of the data
        """

        return pd.DataFrame({'x': x, 'y': y})

    def _export_data(name: str, data: pd.DataFrame):
        """
        Export data from the analysis database.

        Parameters
        ----------
        name : str
                name of the tensor_values to save. Usually this is just the
                analysis name. In the case of species specific analysis,
                this will be further appended to include the name of the
                species.
        data : pd.DataFrame
                Data to be saved.
        Returns
        -------
        Saves a csv file to disc.
        """
        data.to_csv(name)

    def _msd_operation(self, ensemble: tf.Tensor, square: bool = True):
        """
        Perform a simple msd operation.

        Parameters
        ----------
        ensemble : tf.Tensor
            Trajectory over which to compute the msd.
        square : bool
            If true, square the result, else just return the difference.
        Returns
        -------
        msd : tf.Tensor
                Mean square displacement.
        """
        if square:
            return tf.math.squared_difference(tf.gather(ensemble,
                                                        self.tau_values,
                                                        axis=1),
                                              ensemble[:, None, 0])
        else:
            return tf.math.subtract(ensemble, ensemble[:, None, 0])

    def _handle_tau_values(self) -> np.array:
        """
        Handle the parsing of custom tau values.

        Returns
        -------
        times : np.array
            The time values corresponding to the selected tau values
        """
        if isinstance(self.tau_values, int):
            self.data_resolution = self.tau_values
            self.tau_values = np.linspace(0,
                                          self.data_range - 1,
                                          self.tau_values,
                                          dtype=int)
        if isinstance(self.tau_values, list) or isinstance(self.tau_values, np.ndarray):
            self.data_resolution = len(self.tau_values)
            self.data_range = self.tau_values[-1] + 1
        if isinstance(self.tau_values, slice):
            self.tau_values = np.linspace(0,
                                          self.data_range - 1,
                                          self.data_range,
                                          dtype=int)[self.tau_values]
            self.data_resolution = len(self.tau_values)

        times = np.asarray(self.tau_values) * self.experiment.time_step * self.experiment.sample_rate

        return times

    def _prepare_managers(self, data_path: list):
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
        self.memory_manager = MemoryManager(data_path=data_path,
                                            database=self.database,
                                            memory_fraction=0.8,
                                            scale_function=self.scale_function,
                                            gpu=self.gpu)
        self.batch_size, \
        self.n_batches, \
        self.remainder = self.memory_manager.get_batch_size(
            system=self.system_property)

        self.ensemble_loop, minibatch = self.memory_manager.get_ensemble_loop(
            self.data_range, self.correlation_time)
        if minibatch:
            self.batch_size = self.memory_manager.batch_size
            self.n_batches = self.memory_manager.n_batches
            self.remainder = self.memory_manager.remainder

        self.data_manager = DataManager(data_path=data_path,
                                        database=self.database,
                                        data_range=self.data_range,
                                        batch_size=self.batch_size,
                                        n_batches=self.n_batches,
                                        ensemble_loop=self.ensemble_loop,
                                        correlation_time=self.correlation_time,
                                        remainder=self.remainder,
                                        atom_selection=self.atom_selection,
                                        minibatch=minibatch,
                                        atom_batch_size=self.memory_manager.atom_batch_size,
                                        n_atom_batches=self.memory_manager.n_atom_batches,
                                        atom_remainder=self.memory_manager.atom_remainder
                                        )
        self._update_output_signatures()

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
        return f"{self.database_group}_{self.analysis_name}_{self.data_range}_{species}"

    def _save_data(self, name: str, data: pd.DataFrame):
        """
        Save tensor_values to the save tensor_values directory

        Parameters
        ----------
        name : str
                name of the tensor_values to save. Usually this is just the
                analysis name. In the case of species specific analysis,
                this will be further appended to include the name of the
                species.
        data : pd.DataFrame
                Data to be saved.
        """
        # database = AnalysisDatabase(
        #     name=os.path.join(self.experiment.database_path,
        #                       "analysis_database"))
        # database.add_data(name=name, data_frame=data)

    def run_visualization(
            self, x_data, y_data, title: str
    ):
        """
        Run a visualization session on the data.

        Returns
        -------

        """
        self.plot_array.append(self.plotter.construct_plot(
            x_data=x_data, y_data=y_data, title=title, x_label=self.x_label, y_label=self.y_label
        ))

        if self.last_iteration:
            self.plotter.grid_show(self.plot_array)

    def _check_input(self):
        """
        Look for user input that would kill the analysis

        Returns
        -------
        status: int
            if 0, check failed, if 1, check passed.
        """
        if self.data_range > self.experiment.number_of_configurations - \
                self.correlation_time:
            raise ValueError(
                "Data range is impossible for this experiment, "
                "reduce and try again")

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

        popt, pcov = curve_fit(func, log_x[start_index:end_index], log_y[
                                                                   start_index:end_index])

        if 0.85 < popt[0] < 1.15:
            self.loop_condition = True

        else:
            try:
                self.data_range = int(1.1 * self.data_range)
                self.time = np.linspace(0.0,
                                        self.data_range *
                                        self.experiment.time_step *
                                        self.experiment.sample_rate,
                                        self.data_range)

                # end the calculation if the tensor_values range exceeds the relevant bounds
                if self.data_range > self.experiment.number_of_configurations - \
                        self.correlation_time:
                    print("Trajectory not long enough to perform analysis.")
                    raise RangeExceeded
            except RangeExceeded:
                raise RangeExceeded

    def _update_properties_file(self, parameters: dict,
                                delete_duplicate: bool = True):
        """
        Update the experiment properties YAML file.
        """
        log.warning("Using depreciated method `_update_properties_file` \t Please use `update_database` instead.")
        self.update_database(parameters=parameters, delete_duplicate=delete_duplicate)
        #
        #
        #
        # database = PropertiesDatabase(
        #     name=os.path.join(self.experiment.database_path,
        #                       'property_database'))
        # database.add_data(parameters, delete_duplicate)

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
                'Unwrapped_Positions': self._unwrap_choice(), }

            # add the other transformations and merge the dictionaries
            switcher = {**switcher_unwrapping, **switcher_transformations}

            choice = switcher.get(argument,
                                  lambda: "Data not in database and can not "
                                          "be generated.")
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

        if self.loaded_property is None:
            return

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
        Returns
        -------
        Performs the analysis.
        """
        if self.system_property:
            self._calculate_prefactor()

            data_path = [join_path(self.loaded_property, self.loaded_property)]
            self._prepare_managers(data_path)

            batch_generator, batch_generator_args = self.data_manager.batch_generator(
                system=self.system_property)
            batch_data_set = tf.data.Dataset.from_generator(
                generator=batch_generator,
                args=batch_generator_args,
                output_signature=self.batch_output_signature)
            batch_data_set.prefetch(tf.data.experimental.AUTOTUNE)

            for batch_index, batch in enumerate(batch_data_set):
                ensemble_generator, ensemble_generators_args = self.data_manager.ensemble_generator(
                    system=self.system_property)
                ensemble_data_set = tf.data.Dataset.from_generator(
                    generator=ensemble_generator,
                    args=ensemble_generators_args + (batch,),
                    output_signature=self.ensemble_output_signature)

                for ensemble_index, ensemble in tqdm(
                        enumerate(ensemble_data_set),
                        desc="Ensemble Loop",
                        ncols=70,
                        total=self.ensemble_loop):
                    self._apply_operation(ensemble, ensemble_index)

            self._apply_averaging_factor()
            self._post_operation_processes()

        elif self.experimental:
            data_path = [join_path(species,
                                   self.loaded_property) for species
                         in self.experiment.species]
            self._prepare_managers(data_path)
            output = self.run_experimental_analysis()

            return output

        elif self.post_generation:
            self.run_post_generation_analysis()

        else:
            for i, species in enumerate(self.species):
                if i == len(self.species) - 1:
                    self.last_iteration = True
                self._calculate_prefactor(species)

                data_path = [join_path(species, self.loaded_property)]
                self._prepare_managers(data_path)

                batch_generator, batch_generator_args = self.data_manager.batch_generator()
                batch_data_set = tf.data.Dataset.from_generator(
                    generator=batch_generator,
                    args=batch_generator_args,
                    output_signature=self.batch_output_signature)
                batch_data_set = batch_data_set.prefetch(
                    tf.data.experimental.AUTOTUNE)

                for batch_index, batch in tqdm(enumerate(batch_data_set),
                                               ncols=70,
                                               desc=species,
                                               total=self.n_batches,
                                               disable=self.memory_manager.minibatch):

                    ensemble_generator, ensemble_generators_args = self.data_manager.ensemble_generator()
                    ensemble_data_set = tf.data.Dataset.from_generator(
                        generator=ensemble_generator,
                        args=ensemble_generators_args + (batch,),
                        output_signature=self.ensemble_output_signature)
                    ensemble_data_set = ensemble_data_set.prefetch(
                        tf.data.experimental.AUTOTUNE)

                    for ensemble_index, ensemble in enumerate(
                            ensemble_data_set):
                        self._apply_operation(ensemble, ensemble_index)

                self._apply_averaging_factor()
                self._post_operation_processes(species)

            if self.plot:
                plt.legend()
                plt.show()

    def run_experimental_analysis(self):
        """
        For experimental methods
        Returns
        -------

        """
        raise NotImplementedError

    def run_post_generation_analysis(self):
        """
        Run a post-generation analysis.
        """
        raise NotImplementedError

    def run_analysis(self):
        """
        Run the appropriate analysis
        """
        self._check_input()
        self._run_dependency_check()
        if self.experimental:
            log.warning(
                "This is an experimental calculator. Please see the "
                "documentation before using the results.")
        if self.optimize:
            pass
        else:
            return self.perform_computation()

    @property
    def dtype(self):
        """Get the dtype used for the calculator"""
        return self._dtype

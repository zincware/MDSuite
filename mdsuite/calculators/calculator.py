"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Parent class for different analysis

Summary
-------
"""
import logging
import numpy as np
import os
import abc
import random
import sys
import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.axes._subplots import Axes
import tensorflow as tf
import pandas as pd
from scipy.optimize import curve_fit
from mdsuite.utils.exceptions import RangeExceeded
from mdsuite.utils.meta_functions import join_path
from mdsuite.memory_management.memory_manager import MemoryManager
from mdsuite.database.data_manager import DataManager
from mdsuite.database.simulation_database import Database
from mdsuite.calculators.computations_dict import switcher_transformations
from mdsuite.database.properties_database import PropertiesDatabase
from mdsuite.database.analysis_database import AnalysisDatabase
from tqdm import tqdm
from typing import Union

log = logging.getLogger(__file__)


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

    def __init__(self, experiment: object, plot: bool = True, save: bool = True, data_range: int = 500,
                 correlation_time: int = 1, atom_selection: object = np.s_[:], export: bool = True, gpu: bool = False):
        """
        Constructor for the calculator class.

        Parameters
        ----------
        experiment : object
                Experiment object to update.
        plot : bool
                If true, analysis is plotted.
        save : bool
                If true, the analysis is saved.
        data_range : int
                Data range over which to compute.
        correlation_time : int
                Correlation time to use in the analysis.
        atom_selection : np.s_
                Atoms to peform the analysis on.
        export : bool
                If true, analysis results are exported to a csv file.
        gpu : bool
                If true, reduce memory usage to what is allowed on the system GPU.
        """
        # Set upon instantiation of parent class
        self.experiment = experiment
        self.data_range = data_range
        self.plot = plot
        self.save = save
        self.export = export
        self.atom_selection = atom_selection
        self.correlation_time = correlation_time
        self.database = Database(name=os.path.join(self.experiment.database_path, "database.hdf5"))
        self.gpu = gpu
        self.time = np.linspace(0.0, self.data_range * self.experiment.time_step * self.experiment.sample_rate,
                                self.data_range)

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

        self.database_group = None
        self.analysis_name = None

        # Prevent $DISPLAY warnings on clusters.
        if self.experiment.cluster_mode:
            import matplotlib
            matplotlib.use('Agg')

    def update_user_args(self, plot: bool = True, save: bool = True, data_range: int = 500,
                         correlation_time: int = 1, atom_selection: object = np.s_[:], export: bool = True,
                         gpu: bool = False):
        """
        Update the user args that are given by the __call__ method of the calculator

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
        atom_selection : np.s_
                Atoms to peform the analysis on.
        export : bool
                If true, analysis results are exported to a csv file.
        gpu : bool
                If true, reduce memory usage to what is allowed on the system GPU.
        """
        self.data_range = data_range
        self.plot = plot
        self.save = save
        self.export = export
        self.gpu = gpu
        self.correlation_time = correlation_time  # correlation time of the property
        self.atom_selection = atom_selection

        # attributes based on user args
        self.time = np.linspace(0.0, self.data_range * self.experiment.time_step * self.experiment.sample_rate,
                                self.data_range)

    @abc.abstractmethod
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
    def _post_operation_processes(self, species: Union[str, tuple] = None):
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
            dictionary[str.encode(item)] = tf.TensorSpec(shape=(None, None, dimension), dtype=tf.float64)

        return dictionary

    @staticmethod
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

    @staticmethod
    def _export_data(name: str, data: pd.DataFrame):
        """
        Export data from the analysis database.

        Parameters
        ----------
        name : str
                name of the tensor_values to save. Usually this is just the analysis name. In the case of species
                specific analysis, this will be further appended to include the name of the species.
        data : pd.DataFrame
                Data to be saved.
        Returns
        -------
        Saves a csv file to disc.
        """
        data.to_csv(name)

    @staticmethod
    def _msd_operation(ensemble: tf.Tensor, square: bool = True):
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
            return tf.math.squared_difference(ensemble, ensemble[:, None, 0])
        else:
            return tf.math.subtract(ensemble, ensemble[:, None, 0])

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
                                            memory_fraction=0.8,
                                            scale_function=self.scale_function,
                                            gpu=self.gpu)
        self.batch_size, self.n_batches, self.remainder = self.memory_manager.get_batch_size(
            system=self.system_property)

        self.ensemble_loop, minibatch = self.memory_manager.get_ensemble_loop(self.data_range, self.correlation_time)
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
                In the case of a species specific analysis, make sure a species is put here. Otherwise, it is set to
                System.
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
                name of the tensor_values to save. Usually this is just the analysis name. In the case of species
                specific analysis, this will be further appended to include the name of the species.
        data : pd.DataFrame
                Data to be saved.
        """
        database = AnalysisDatabase(name=os.path.join(self.experiment.database_path, "analysis_database"))
        database.add_data(name=name, data_frame=data)

        """
        title = '_'.join([title, str(self.data_range)])
        with hf.File(os.path.join(self.experiment.database_path, 'analysis_data.hdf5'), 'r+') as db:
            if title in db[self.database_group].keys():
                del db[self.database_group][title]
                db[self.database_group].create_dataset(title, data=data, dtype=float)
            else:
                db[self.database_group].create_dataset(title, data=data, dtype=float)
        """

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

    def _update_properties_file(self, parameters: dict, delete_duplicate: bool = True):
        """
        Update the experiment properties YAML file.
        """
        database = PropertiesDatabase(name=os.path.join(self.experiment.database_path, 'property_database'))
        database.add_data(parameters, delete_duplicate)

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

            # add the other transformations and merge the dictionaries
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
            batch_generator, batch_generator_args = self.data_manager.batch_generator(system=self.system_property)
            batch_data_set = tf.data.Dataset.from_generator(generator=batch_generator,
                                                            args=batch_generator_args,
                                                            output_signature=self.batch_output_signature)
            batch_data_set.prefetch(tf.data.experimental.AUTOTUNE)
            for batch_index, batch in enumerate(batch_data_set):
                ensemble_generator, ensemble_generators_args = self.data_manager.ensemble_generator(
                    system=self.system_property)
                ensemble_data_set = tf.data.Dataset.from_generator(generator=ensemble_generator,
                                                                   args=ensemble_generators_args + (batch,),
                                                                   output_signature=self.ensemble_output_signature)
                for ensemble_index, ensemble in tqdm(enumerate(ensemble_data_set), desc="Ensemble Loop", ncols=70,
                                                     total=self.ensemble_loop):
                    self._apply_operation(ensemble, ensemble_index)

            self._apply_averaging_factor()
            self._post_operation_processes()

        elif self.experimental:
            data_path = [join_path(species, self.loaded_property) for species in self.experiment.species]
            self._prepare_managers(data_path)
            output = self.run_experimental_analysis()

            return output

        elif self.post_generation:
            self.run_post_generation_analysis()

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
                for batch_index, batch in tqdm(enumerate(batch_data_set), ncols=70, desc=species, total=self.n_batches,
                                               disable=self.memory_manager.minibatch):
                    ensemble_generator, ensemble_generators_args = self.data_manager.ensemble_generator()
                    ensemble_data_set = tf.data.Dataset.from_generator(generator=ensemble_generator,
                                                                       args=ensemble_generators_args + (batch,),
                                                                       output_signature=self.ensemble_output_signature)
                    ensemble_data_set = ensemble_data_set.prefetch(tf.data.experimental.AUTOTUNE)
                    for ensemble_index, ensemble in enumerate(ensemble_data_set):
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
            log.warning("\n ########## \n "
                        "This is an experimental calculator. It is provided as it can still be used, however, "
                        "it may not be"
                        " memory safe or completely accurate. \n Please see the documentation for more information. \n "
                        "#########")
        if self.optimize:
            pass
        else:
            return self.perform_computation()

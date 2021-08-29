"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Module for the Project class

Summary
-------
Module containing all the code for the Project class. The project class is the governing class in the
mdsuite program. Within the project class include all of the method required to add a new experiment and
compare the results of the analysis on that experiment.
"""
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Union
import shutil
from mdsuite.experiment import Experiment
from mdsuite.utils.meta_functions import simple_file_read, find_item
from mdsuite.database.project_database import ProjectDatabase
import mdsuite.database.scheme as db

log = logging.getLogger(__file__)


class Project(ProjectDatabase):
    """
    Class for the main container of all experiments.

    The Project class acts as the encompassing class for analysis with MDSuite. It contains all method required to add
    and analyze new experiments. These experiments may then be compared with one another quickly. The state of the class
    is saved and updated after each operation in order to retain the most current state of the analysis.

    Attributes
    ----------
    name : str
            The name of the project

    description : str
            A short description of the project

    storage_path : str
            Where to store the tensor_values and databases. This may not simply be the current directory if the
            databases are expected to be quite large.

    experiments : dict
            A dict of class objects. Class objects are instances of the experiment class for different
            experiments.
    """

    def __init__(self, name: str = None, storage_path: str = "./", description: str = None):
        """
        Project class constructor

        The constructor will check to see if the project already exists, if so, it will load the state of each of the
        classes so that they can be used again. If the project is new, the constructor will build the necessary file
        structure for the project.

        Parameters
        ----------
        name : str
                The name of the project.
        storage_path : str
                Where to store the tensor_values and databases. This should be a place with sufficient storage space
                for the full analysis.
        """
        super().__init__()
        if name is None:
            self.name = f"MDSuite_Project"
        else:
            self.name = f"MDSuite_Project_{name}"
        self.storage_path = storage_path

        # Properties
        self._experiments = {}

        # Check for project directory, if none exist, create a new one
        project_dir = Path(f"{self.storage_path}/{self.name}")
        if project_dir.exists():
            log.info("Loading the class state")
            log.info(f"Available experiments are: {self.db_experiments}")
        else:
            project_dir.mkdir(parents=True, exist_ok=True)

        self.build_database()

        # Database Properties
        self.description = description

    def __str__(self):
        """

        Returns
        -------
        str:
            A list of all available experiments like "1.) Exp01\n2.) Exp02\n3.) Exp03"

        """
        return "\n".join([f"{exp.id}.) {exp.name}" for exp in self.db_experiments])

    def add_experiment(self, experiment: str = None, timestep: float = None, temperature: float = None,
                       units: str = None, cluster_mode: bool = None, active: bool = True):
        """
        Add an experiment to the project

        Parameters
        ----------
        active: bool, default = True
                Activate the experiment when added
        cluster_mode : bool
                If true, cluster mode is parsed to the experiment class.
        experiment : str
                Name to use for the experiment class.
        timestep : float
                Timestep used during the simulation.
        temperature : float
                Temperature the simulation was performed at and is to be used in calculation.
        units : str
                LAMMPS units used
        """

        if experiment is None:
            experiment = f"Experiment_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            # set the experiment name to the current date and time if None is provided

        # Run a query to see if that experiment already exists
        with self.session as ses:
            experiments = ses.query(db.Experiment).filter(db.Experiment.name == experiment).all()
        if len(experiments) > 0:
            log.info("This experiment already exists")
            self.load_experiments(experiment)
            return

        # If the experiment does not exists, instantiate a new Experiment
        new_experiment = Experiment(
            project=self,
            experiment_name=experiment,
            storage_path=f"{self.storage_path}/{self.name}",
            time_step=timestep,
            units=units,
            temperature=temperature,
            cluster_mode=cluster_mode
        )

        new_experiment.active = active

        # # TODO should be replaced by active experiments that are loaded!
        # self.experiments[new_experiment.analysis_name] = new_experiment  # add the new experiment to the dictionary

    def load_experiments(self, names: Union[str, list]):
        """Load experiments, such that they are used for the computations

        Parameters
        ----------
        names: Name or list of names of experiments that should be instantiated and loaded into self.experiments

        Returns
        -------

        """

        if isinstance(names, str):
            names = [names]

        for name in names:
            if name not in [exp.name for exp in self.db_experiments]:
                raise ValueError(f'Could not find an experiment titled {name}!')

            new_experiment = Experiment(
                project=self,
                experiment_name=name,
            )

            new_experiment.active = True

            # self.experiments = {new_experiment.analysis_name: new_experiment, **self.experiments}
            # merge two dicts - this will be removed eventually

    def add_data(self, data_sets: dict, file_format='lammps_traj'):
        """
        Add data to an experiment. This is a method so that parallelization is possible amongst data addition to
        different experiments at the same time.

        Parameters
        ----------
        data_sets: dict
            Dictionary containing the name of the experiment as key and the data path as value
        file_format: dict or str
            Dictionary containing the name of the experiment as key and the file_format as value.
            Alternativly only a string of the file_format if all files have the same format.

        Returns
        -------
        Updates the experiment classes.
        """
        if isinstance(file_format, dict):
            try:
                assert file_format.keys() == data_sets.keys()
            except AssertionError:
                log.error("Keys of the data_sets do not match keys of the file_format")

            for item in data_sets:
                self.experiments[item].add_data(data_sets[item], file_format=file_format[item])
        else:
            for item in data_sets:
                self.experiments[item].add_data(data_sets[item], file_format=file_format)

    # def get_results(self, key_to_find):
    #     """
    #     Gets the results from the experiments and puts them in a dict
    #
    #     Parameters
    #     ----------
    #     key_to_find : str
    #         name of the parameter to search in the results.
    #
    #     Returns
    #     -------
    #     results: dict
    #         collects the results from the different experiments
    #     """
    #
    #     results = {}
    #     for experiment_name, experiment_class in self.experiments.items():
    #         results_yaml = experiment_class.results  # this is a dict with the results from the yaml file
    #         result = find_item(results_yaml, key_to_find)
    #
    #         if isinstance(result, str):
    #             if result.startswith('['):
    #                 result = list(result.replace('[', '').replace(']', '').split(','))
    #             else:
    #                 result = float(result)
    #
    #         if isinstance(result, list):
    #             result = [float(res) for res in result]  # convert results to floats
    #         results[experiment_name] = result
    #
    #     return results

    # def get_properties(self, parameters: dict, experiments: list = None):
    #     """
    #     Get some property of each experiment.
    #
    #     Parameters
    #     ----------
    #     parameters : dict
    #             Parameters to be used in the addition, i.e.
    #
    #             .. code-block:: python
    #
    #                {"Analysis": "Green_Kubo_Self_Diffusion",  "Subject": "Na", "data_range": 500}
    #
    #     experiments : list
    #             List of experiments to fetch information for. If None, all will be searched.
    #
    #     Returns
    #     -------
    #     properties_dict : dict
    #             A dictionary of lists of properties for each system
    #     """
    #     if experiments is None:
    #         experiments = list(self.experiments)
    #
    #     properties_dict = {}
    #     for item in experiments:
    #         properties_dict[item] = self.experiments[item].export_property_data(parameters.copy())
    #
    #     return properties_dict

    # def get_attribute(self, attribute):
    #     """
    #     Get an attribute from the experiments. Equivalent to get_results but for system parameters such as:
    #     temperature, time_step, etc.
    #
    #     Parameters
    #     ----------
    #     attribute : str
    #         name of the parameter to search in the experiment.
    #
    #     Returns
    #     -------
    #     results: dict
    #         collects the results from the different experiments
    #     """
    #
    #     results = {}
    #     for experiment_name, experiment_class in self.experiments.items():
    #         value_attr = experiment_class.__getattribute__(attribute)
    #         results[experiment_name] = value_attr
    #
    #     return results

    # def remove_experiment(self, experiment_name: str):
    #     """
    #     Delete an experiment from the project
    #     Parameters
    #     ----------
    #     experiment_name
    #
    #     Returns
    #     -------
    #     Updates the class state.
    #     """
    #     if experiment_name not in list(self.experiments):
    #         print("Experiment does not exist")
    #         return
    #     else:
    #         try:
    #             dir_path = os.path.join(self.storage_path, self.name, experiment_name)
    #             shutil.rmtree(dir_path)
    #             self.experiments.pop(experiment_name, None)
    #             # self._save_class()
    #         except InterruptedError:
    #             print("You are likely using a notebook of some kind such as jupyter. Please restart the kernel and try"
    #                   "to do this again.")

    @property
    def experiments(self) -> dict:
        """Get a dict of instantiated experiments that are currently selected!"""
        # TODO there could be a performance increase if the experiments are stored instead of instantiated every time
        #   this property is called.
        experiments = {}

        with self.session as ses:
            db_experiments = ses.query(db.Experiment).filter(db.Experiment.active).all()

        for exp in db_experiments:
            exp: db.Experiment
            experiments[exp.name] = Experiment(project=self, experiment_name=exp.name)

        return experiments

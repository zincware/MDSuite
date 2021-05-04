"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.
"""

"""
Module for the Project class

Summary
-------
Module containing all the code for the Project class. The project class is the governing class in the
mdsuite program. Within the project class include all of the method required to add a new experiment and
compare the results of the analysis on that experiment.
"""

import os
import pickle
from datetime import datetime
from pathlib import Path

from typing import Union
import shutil
from mdsuite.experiment.experiment import Experiment
from mdsuite.utils.meta_functions import simple_file_read, find_item


class Project:
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
            Where to store the tensor_values and databases. This may not simply be the current direcotry if the databases are
            expected to be quite large.

    experiments : dict
            A dict of class objects. Class objects are instances of the experiment class for different
            experiments.
    """

    def __init__(self, name: str = "My_Project", storage_path: str = "./"):
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
                Where to store the tensor_values and databases. This should be a place with sufficient storage space for the
                full analysis.
        """

        self.name = f"{name}_MDSuite_Project"  # the name of the project
        self.description = None  # A short description of the project
        self.storage_path = storage_path  # Tell me where to store this tensor_values

        self.experiments = {}  # type: dict[str, Experiment]  #experiments added to this project

        # Check for project directory, if none exist, create a new one
        test_dir = Path(f"{self.storage_path}/{self.name}")
        if test_dir.exists():
            print("Loading the class state")
            self._load_class()  # load the class state

            # load the class state for each experiment attached to the Project.
            for experiment in self.experiments.values():
                experiment.load_class()

            # List the experiments available to the user
            self.__str__()
        else:
            os.mkdir(f"{self.storage_path}/{self.name}")  # create a new directory for the project
            self._save_class()  # Save the initial class state

    def __str__(self):
        return self.list_experiments()

    def add_description(self, description: str):
        """
        Allow users to add a short description to their project

        Parameters
        ----------
        description : str
                Description of the project. If the string ends in .txt, the contents of the txt file will be read. If
                it ends in .md, same outcome. Anything else will be read as is.
        """

        # Check the file type and read accordingly
        if description[-3:] == "txt":
            self.description = simple_file_read(description)
        elif description[-2:] == "md":
            self.description = simple_file_read(description)
        else:
            self.description = description

        self._save_class()  # Update the class state

    def list_experiments(self):
        """
        List the available experiments as a numerical list.

        Returns
        -------
        str: per-line list of experiments
        """
        list_experiments = []
        for idx, experiment in enumerate(self.experiments):
            list_experiments.append(f"{idx}.) {experiment}")
        return '\n'.join(list_experiments)

    def _load_class(self):
        """
        Load the class state of the Project class from a saved file.
        """

        class_file = open(f'{self.storage_path}/{self.name}/{self.name}_state.bin', 'rb')  # Open the class state file
        pickle_data = class_file.read()  # Read in the tensor_values
        class_file.close()  # Close the state file

        self.__dict__.update(pickle.loads(pickle_data))  # Initialize the class with the loaded parameters

    def _save_class(self):
        """
        Saves class instance

        In order to keep properties of a class the state must be stored. This method will store the instance of the
        class for later re-loading
        """
        self.__dict__.update(self.experiments)  # updates the dictionary with the new experiments added

        save_file = open(f"{self.storage_path}/{self.name}/{self.name}_state.bin", 'wb')  # Open the class state file
        save_file.write(pickle.dumps(self.__dict__))  # write the current state of the class
        save_file.close()  # Close the state file

    def add_experiment(self, experiment: Union[str, dict] = None, timestep: float = None, temperature: float = None,
                       units: str = None, cluster_mode: bool = None):
        """
        Add an experiment to the project

        Parameters
        ----------
        cluster_mode : bool
                If true, cluster mode is parsed to the experiment class.
        experiment_name : str
                Name to use for the experiment class.
        timestep : float
                Timestep used during the simulation.
        temperature : float
                Temperature the simulation was performed at and is to be used in calculation.
        units : str
                LAMMPS units used
        """

        if type(experiment) is str:
            # Set a name in case none is given
            if experiment is None:
                experiment = f"Experiment_{datetime.now()}"  # set the experiment name to the current date and time

            # Run a check to see if that experiment already exists
            test_file = Path(f"{self.storage_path}/{self.name}/{experiment}")

            # Check if the file exists, if so, return the method without changing the class state
            if test_file.exists():
                print("This experiment already exists")
                return

            # If the experiment does not exists, instantiate a new Experiment
            new_experiment = Experiment(experiment,
                                        storage_path=f"{self.storage_path}/{self.name}",
                                        time_step=timestep,
                                        units=units,
                                        temperature=temperature,
                                        cluster_mode=cluster_mode)

            self.experiments[new_experiment.analysis_name] = new_experiment  # add the new experiment to the dictionary
        else:
            for item in experiment:
                # Run a check to see if that experiment already exists
                test_file = Path(f"{self.storage_path}/{self.name}/{item}")

                # Check if the file exists, if so, return the method without changing the class state
                if test_file.exists():
                    print("This experiment already exists, aborting addition")
                    continue

                # If the experiment does not exists, instantiate a new Experiment
                new_experiment = Experiment(item,
                                            storage_path=f"{self.storage_path}/{self.name}",
                                            **experiment[item])

                self.experiments[
                    new_experiment.analysis_name] = new_experiment  # add the new experiment to the dictionary

        self._save_class()  # Save the class state

    def add_data(self, data_sets: dict):
        """
        Add data to an experiment. This is a method so that parallelization is possible amongst data addition to
        different experiments at the same time.

        Returns
        -------
        Updates the experiment classes.
        """
        for item in data_sets:
            self.experiments[item].add_data(data_sets[item])

    def get_results(self, key_to_find):
        """
        Gets the results from the experiments and puts them in a dict

        Parameters
        ----------
        key_to_find : str
            name of the parameter to search in the results.

        Returns
        -------
        results: dict
            collects the results from the different experiments
        """

        results = {}
        for experiment_name, experiment_class in self.experiments.items():
            results_yaml = experiment_class.results  # this is a dict with the results from the yaml file
            result = find_item(results_yaml, key_to_find)

            if isinstance(result, str):
                if result.startswith('['):
                    result = list(result.replace('[', '').replace(']', '').split(','))
                else:
                    result = float(result)

            if isinstance(result, list):
                result = [float(res) for res in result] # convert results to floats
            results[experiment_name] = result

        return results

    def get_properties(self, parameters: dict, experiments: list = None):
        """
        Get some property of each experiment.

        Parameters
        ----------
        parameters : dict
                Parameters to be used in the addition, i.e.
                {"Analysis": "Green_Kubo_Self_Diffusion",
                 "Subject": "Na",
                 "data_range": 500}
        experiments : list
                List of experiments to fetch information for. If None, all will be searched.

        Returns
        -------
        properties_dict : dict
                A dictionary of lists of properties for each system
        """
        if experiments is None:
            experiments = list(self.experiments)

        properties_dict = {}
        for item in experiments:
            properties_dict[item] = self.experiments[item].export_property_data(parameters.copy())

        return properties_dict

    def get_attribute(self, attribute):
        """
        Get an attribute from the experiments. Equivalent to get_results but for system parameters such as:
        temperature, time_step, etc.

        Parameters
        ----------
        attribute : str
            name of the parameter to search in the experiment.

        Returns
        -------
        results: dict
            collects the results from the different experiments
        """

        results = {}
        for experiment_name, experiment_class in self.experiments.items():
            value_attr = experiment_class.__getattribute__(attribute)  # this is a dict with the results from the yaml file
            results[experiment_name] = value_attr

        return results

    def remove_experiment(self, experiment_name: str):
        """
        Delete an experiment from the project
        Parameters
        ----------
        experiment_name

        Returns
        -------
        Updates the class state.
        """
        if experiment_name not in list(self.experiments):
            print("Experiment does not exist")
            return
        else:
            try:
                dir_path = os.path.join(self.storage_path, self.name, experiment_name)
                shutil.rmtree(dir_path)
                self.experiments.pop(experiment_name, None)
                self._save_class()
            except InterruptedError:
                print("You are likely using a notebook of some kind such as jupyter. Please restart the kernel and try"
                      "to do this again.")

"""
Authors: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com

Summary
-------
Module containing all the code for the Project class. The project class is the governing class in the
mdsuite program. Within the project class include all of the method required to add a new experiment and
compare the results of the analysis on that experiment.
"""

import os
from datetime import datetime
import numpy as np

from pathlib import Path
import pickle

from mdsuite.experiment.experiment import Experiment
from mdsuite.utils.meta_functions import simple_file_read


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
            Where to store the data and databases. This may not simply be the current direcotry if the databases are
            expected to be quite large.

    experiments : list
            A list of class objects. Class objects are instances of the experiment class for different
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
                Where to store the data and databases. This should be a place with sufficient storage space for the
                full analysis.
        """

        self.name = f"{name}_MDSuite_Project"  # the name of the project
        self.description = None  # A short description of the project
        self.storage_path = storage_path  # Tell me where to store this data

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
            print("List of available experiments")
            self.list_experiments()
        else:
            os.mkdir(f"{self.storage_path}/{self.name}")  # create a new directory for the project
            self._save_class()  # Save the initial class state

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
            self.description = "".join(np.array(simple_file_read(description)).flatten())
        elif description[-2:] == "md":
            self.description = "".join(np.array(simple_file_read(description)).flatten())
        else:
            self.description = description

        self._save_class()  # Update the class state

    def list_experiments(self):
        """
        List the available experiments as a numerical list.
        """

        counter = 0
        for experiment in self.experiments:
            print(f"{counter}.) {experiment}")

    def _load_class(self):
        """
        Load the class state of the Project class from a saved file.
        """

        class_file = open(f'{self.storage_path}/{self.name}/{self.name}_state.bin', 'rb')  # Open the class state file
        pickle_data = class_file.read()  # Read in the data
        class_file.close()  # Close the state file

        self.__dict__ = pickle.loads(pickle_data)  # Initialize the class with the loaded parameters

    def _save_class(self):
        """
        Saves class instance

        In order to keep properties of a class the state must be stored. This method will store the instance of the
        class for later re-loading
        """

        save_file = open(f"{self.storage_path}/{self.name}/{self.name}_state.bin", 'wb')  # Open the class state file
        save_file.write(pickle.dumps(self.__dict__))  # write the current state of the class
        save_file.close()  # Close the state file

    def add_experiment(self, experiment_name: str=None, timestep: float = None, temperature: float = None,
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

        # Set a name in case none is given
        if experiment_name is None:
            experiment_name = f"Experiment_{datetime.now()}"  # set the experiment name to the current date and time

        # Run a check to see if that experiment already exists
        test_file = Path(f"{self.storage_path}/{self.name}/{experiment_name}")

        # Check if the file exists, if so, return the method without changing the class state
        if test_file.exists():
            print("This experiment already exists, aborting addition")
            return

        # If the experiment does not exists, instantiate a new Experiment
        new_experiment = Experiment(experiment_name,
                                    storage_path=f"{self.storage_path}/{self.name}",
                                    time_step=timestep,
                                    units=units,
                                    temperature=temperature,
                                    cluster_mode=cluster_mode)

        self.experiments[new_experiment.analysis_name] = new_experiment  # add the new experiment to the dictionary

        self._save_class()  # Save the class state

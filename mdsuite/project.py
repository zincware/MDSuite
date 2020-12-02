"""
Authors: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com

Description: Module containing all the code for the Project class. The project class is the governing class in the
             mdsuite program.
"""

import os
import sys
from datetime import datetime
import numpy as np

from pathlib import Path
import pickle

from mdsuite.experiment.experiment import Experiment
from mdsuite.utils.meta_functions import simple_file_read


class Project:
    """ Class for the main container of all experiments

    Attributes:
        name (str) -- The name of the project
        description (str) --  A short description of the project
        storage_path (str) -- Where to store the data and databases

        experiments (list) -- A list of class objects. Class objects are instances of the experiment class for different
                              experiments.
    """

    def __init__(self, name="My_Project", storage_path="./"):
        """ Standard constructor """

        self.name = f"{name}_MDSuite_Project"  # the name of the project
        self.description = None  # A short description of the project
        self.storage_path = storage_path  # Tell me where to store this data

        self.experiments = {}  # experiments added to this project

        # Check for project directory, if none exist, create a new one
        test_dir = Path(f"{self.storage_path}/{self.name}")
        if test_dir.exists():
            print("Loading the class state")
            self._load_class()  # load the class state

            print("List of available experiments")
            self.list_experiments()
        else:
            os.mkdir(f"{self.storage_path}/{self.name}")  # create a new directory for the project
            self._save_class()  # Save the initial class state

    def add_description(self, description):
        """ Allow users to add a short description to their project

        :argument description (str) -- description to be used. If the string ends in .txt, the contents of the txt file
                                       will be read. If it ends in .md, same outcome. Anything else will be read as is.
        """

        if description[-3:] is "txt":
            self.description = "".join(np.array(simple_file_read(description)).flatten())
        elif description[-2:] is "md":
            self.description = "".join(np.array(simple_file_read(description)).flatten())
        else:
            self.description = description

        self._save_class()

    def list_experiments(self):
        """ List the available experiments """

        counter = 0
        for experiment in self.experiments:
            print(f"{counter}.) {experiment}")

    def _load_class(self):
        """ Load the class state """

        class_file = open(f'{self.storage_path}/{self.name}/{self.name}_state.bin', 'rb')
        pickle_data = class_file.read()
        class_file.close()

        self.__dict__ = pickle.loads(pickle_data)

    def _save_class(self):
        """ Saves class instance

        In order to keep properties of a class the state must be stored. This method will store the instance of the
        class for later re-loading
        """

        save_file = open(f"{self.storage_path}/{self.name}/{self.name}_state.bin", 'wb')
        save_file.write(pickle.dumps(self.__dict__))
        save_file.close()

    def add_experiment(self, experiment_name=None, timestep=None, temperature=None, units=None):
        """ add an experiment to the project """

        # Set a name in case none is given
        if experiment_name is None:
            experiment_name = f"Experiment_{datetime.now()}"  # set the experiment name to the current date and time

        # Run a check to see if that experiment already exists
        test_file = Path(f"{self.storage_path}/{self.name}/{experiment_name}")
        if test_file.exists():
            print("This experiment already exists, aborting addition")
            sys.exit()

        new_experiment = Experiment(experiment_name,
                                    storage_path=f"{self.storage_path}/{self.name}",
                                    timestep=timestep,
                                    units=units,
                                    temperature=temperature)

        self.experiments[new_experiment.analysis_name] = new_experiment  # add the new experiment to the dictionary

        self._save_class()  # Save the class state

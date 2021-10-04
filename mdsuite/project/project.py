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
from datetime import datetime
from pathlib import Path
from typing import Union
from mdsuite.utils.meta_functions import DotDict
from mdsuite.calculators import RunComputation
from mdsuite.database.project_database import ProjectDatabase
import mdsuite.database.scheme as db
from mdsuite.experiment import Experiment
from mdsuite.utils import Units

from typing import Dict

log = logging.getLogger(__name__)


class Project(ProjectDatabase):
    """
    Class for the main container of all experiments.

    The Project class acts as the encompassing class for analysis with MDSuite.
    It contains all method required to add and analyze new experiments. These
    experiments may then be compared with one another quickly. The state of the
    class is saved and updated after each operation in order to retain the
    most current state of the analysis.

    Attributes
    ----------
    name : str
            The name of the project

    description : str
            A short description of the project

    storage_path : str
            Where to store the tensor_values and databases. This may not simply
            be the current directory if the databases are expected to be
            quite large.
    experiments : dict
            A dict of class objects. Class objects are instances of the experiment class
            for different experiments.
    """

    def __init__(
        self, name: str = None, storage_path: str = "./", description: str = None
    ):
        """
        Project class constructor

        The constructor will check to see if the project already exists, if so,
        it will load the state of each of the classes so that they can be used
        again. If the project is new, the constructor will build the necessary
        file structure for the project.

        Parameters
        ----------
        name : str
                The name of the project.
        storage_path : str
                Where to store the tensor_values and databases. This should be
                a place with sufficient storage space for the full analysis.
        """
        super().__init__()
        if name is None:
            self.name = f"MDSuite_Project"
        else:
            self.name = name
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

    def add_experiment(
        self,
        experiment: str = None,
        timestep: float = None,
        temperature: float = None,
        units: Union[str, Units] = None,
        cluster_mode: bool = None,
        active: bool = True,
        data: Union[str, list, dict] = None,
    ):
        """
        Add an experiment to the project

        Parameters
        ----------
        data:
            data that should be added to the experiment. If dict,
            {file:<file>, format:<format>}
        active: bool, default = True
                Activate the experiment when added
        cluster_mode : bool
                If true, cluster mode is parsed to the experiment class.
        experiment : str
                Name to use for the experiment class.
        timestep : float
                Timestep used during the simulation.
        temperature : float
                Temperature the simulation was performed at and is to be used
                in calculation.
        units : str
                LAMMPS units used
        """

        if experiment is None:
            experiment = f"Experiment_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            # set the experiment name to the current date and time if None is provided

        # Run a query to see if that experiment already exists
        with self.session as ses:
            experiments = (
                ses.query(db.Experiment).filter(db.Experiment.name == experiment).all()
            )
        if len(experiments) > 0:
            log.info("This experiment already exists")
            self.load_experiments(experiment)
            return

        # If the experiment does not exists, instantiate a new Experiment
        new_experiment = Experiment(
            project=self,
            experiment_name=experiment,
            time_step=timestep,
            units=units,
            temperature=temperature,
            cluster_mode=cluster_mode,
        )

        new_experiment.active = active

        # Update the internal experiment dictionary for self.experiment property
        self._experiments[experiment] = new_experiment

        if data is not None:

            def handle_file_format(inp):
                """Run experiment.add_data

                Parameters
                ----------
                inp: str, dict, list
                    If dict, {file:<file>, format:<format>}
                """
                if isinstance(inp, str):
                    self.experiments[experiment].add_data(trajectory_file=inp)
                if isinstance(inp, dict):
                    try:
                        self.experiments[experiment].add_data(
                            trajectory_file=inp["file"], file_format=inp["format"]
                        )
                    except KeyError:
                        raise KeyError(
                            "passed dictionary does not contain `file` and `format`,"
                            f" but {[x for x in inp]}"
                        )
                if isinstance(inp, list):
                    for obj in inp:
                        handle_file_format(obj)

            handle_file_format(data)

    def load_experiments(self, names: Union[str, list]):
        """Alias for activate_experiments"""
        self.activate_experiments(names)

    def activate_experiments(self, names: Union[str, list]):
        """Load experiments, such that they are used for the computations

        Parameters
        ----------
        names: Name or list of names of experiments that should be instantiated
               and loaded into self.experiments.

        Returns
        -------
        Updates the class state.
        """

        if isinstance(names, str):
            names = [names]

        for name in names:
            self.experiments[name].active = True

    def disable_experiments(self, names: Union[str, list]):
        """Disable experiments

        Parameters
        ----------
        names: Name or list of names of experiments that should be instantiated
               and loaded into self.experiments

        Returns
        -------

        """

        if isinstance(names, str):
            names = [names]

        for name in names:
            self.experiments[name].active = False

    def add_data(self, data_sets: dict, file_format="lammps_traj"):
        """
        Add data to an experiment. This is a method so that parallelization is
        possible amongst data addition to different experiments at the same
        time.

        Parameters
        ----------
        data_sets: dict
            Dictionary containing the name of the experiment as key and the
            data path as value
        file_format: dict or str
            Dictionary containing the name of the experiment as key and the
            file_format as value. Alternativly only a string of the
            file_format if all files have the same format.

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
                self.experiments[item].add_data(
                    data_sets[item], file_format=file_format[item]
                )
        else:
            for item in data_sets:
                self.experiments[item].add_data(
                    data_sets[item], file_format=file_format
                )

    @property
    def run(self) -> RunComputation:
        """Method to access the available calculators

        Returns
        -------
        RunComputation:
            class that has all available calculators as properties
        """
        return RunComputation(experiments=[x for x in self.experiments.values()])

    @property
    def experiments(self) -> Dict[str, Experiment]:
        """Get a DotDict of instantiated experiments!

        Examples
        --------
        Either use
            >>> Project().experiments.<experiment_name>.run_compution.<Computation>
        Or use
            >>> Project.experiments["<experiment_name"].run_computation.<Computation>
        """

        with self.session as ses:
            db_experiments = ses.query(db.Experiment).all()

        for exp in db_experiments:
            exp: db.Experiment
            if exp.name not in self._experiments:
                self._experiments[exp.name] = Experiment(
                    project=self, experiment_name=exp.name
                )

        return DotDict(self._experiments)

    @property
    def active_experiments(self) -> Dict[str, Experiment]:
        """Get a DotDict of instantiated experiments that are currently selected!"""

        active_experiment = {
            key: val for key, val in self._experiments.items() if val.active
        }

        return DotDict(active_experiment)

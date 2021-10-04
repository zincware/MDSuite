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
import logging
from pathlib import Path
from mdsuite.calculators import RunComputation
from mdsuite.time_series import time_series_dict
from mdsuite.transformations.transformation_dict import transformations_dict
from mdsuite.utils.units import units_dict, Units
from mdsuite.database.experiment_database import ExperimentDatabase
from mdsuite.visualizer.trajectory_visualizer import SimulationVisualizer

from .add_files import ExperimentAddingFiles
from .run_module import RunModule

from typing import Union

log = logging.getLogger(__name__)


class Experiment(ExperimentDatabase, ExperimentAddingFiles):
    """
    The central experiment class fundamental to all analysis.

    Attributes
    ----------
    name : str
            The name of the analysis being performed e.g. NaCl_1400K
    storage_path : str
            Path to where the tensor_values should be stored (best to have  drive
            capable of storing large files)
    temperature : float
            The temperature of the simulation that should be used in some analysis.
            Necessary as it cannot be easily read in from the simulation tensor_values.
    time_step : float
            Time step of the simulation e.g 0.002. Necessary as it cannot be easily
            read in from the trajectory.
    volume : float
            Volume of the simulation box
    species : dict
            A dictionary of the species in the experiment and their properties.
            Their properties includes index location in the trajectory file, mass of
            the species as taken from the PubChem database_path, and the charge taken
            from the same database_path. When using these properties, it is best that
            users confirm this information, with exception to the indices as they are
            read from the file and will be correct.
    number_of_atoms : int
            The total number of atoms in the simulation
    """

    def __init__(
        self,
        project,
        experiment_name,
        time_step=None,
        temperature=None,
        units: Union[str, Units] = None,
        cluster_mode=False,
    ):
        """
        Initialise the experiment class.

        Attributes
        ----------
        experiment_name : str
                The name of the analysis being performed e.g. NaCl_1400K
        storage_path : str
                Path to where the tensor_values should be stored (best to have  drive
                capable of storing large files)
        temperature : float
                The temperature of the simulation that should be used in some analysis.
        time_step : float
                Time step of the simulation e.g 0.002. Necessary as it cannot be easily
                read in from the trajectory.
        units: Union[str, dict], default = "real"
            The units to be used in the experiment to convert to SI
        cluster_mode : bool
                If true, several parameters involved in plotting and parallelization
                will be adjusted so as to allow for optimal performance on a large
                computing cluster.
        """

        # Taken upon instantiation
        super().__init__(project=project, experiment_name=experiment_name)
        self.name = experiment_name
        self.storage_path = Path(project.storage_path, project.name).as_posix()
        self.cluster_mode = cluster_mode

        # ExperimentDatabase stored properties:
        # ------- #
        # set default values
        if self.number_of_configurations is None:
            self.number_of_configurations = 0
        # update database (None values are ignored)
        self.temperature = temperature
        self.time_step = time_step

        # Available properties that aren't set on default
        self.number_of_atoms = None
        # ------- #

        # Added from trajectory file

        if self.units is None:
            if units is None:
                units = "real"
            self.units = self.units_to_si(units)  # Units used during the simulation.

        self.box_array = None  # Box vectors.
        self.dimensions = None  # Dimensionality of the experiment.

        self.sample_rate = (
            None  # Rate at which configurations are dumped in the trajectory.
        )
        self.batch_size = None  # Number of configurations in each batch.
        self.volume = None
        self.properties = None  # Properties measured in the simulation.
        self.property_groups = (
            None  # Names of the properties measured in the simulation
        )

        # Internal File paths
        self.experiment_path: str
        self.database_path: str
        self.figures_path: str
        self.logfile_path: str
        self._create_internal_file_paths()  # fill the path attributes

        # Memory properties
        self.memory_requirements = (
            {}
        )  # TODO I think this can be removed. - Not until all calcs are updated.

        # Check if the experiment exists and load if it does.
        self._load_or_build()

        self.analyse_time_series = RunModule(self, time_series_dict)

    @property
    def run(self) -> RunComputation:
        """Method to access the available calculators

        Returns
        -------
        RunComputation:
            class that has all available calculators as properties
        """
        return RunComputation(experiment=self)

    def __repr__(self):
        return f"exp_{self.name}"

    def _create_internal_file_paths(self):
        """
        Create or update internal file paths
        """
        self.experiment_path = Path(
            self.storage_path, self.name
        )  # path to the experiment files
        self.database_path = Path(
            self.experiment_path, "databases"
        )  # path to the databases
        self.figures_path = Path(
            self.experiment_path, "figures"
        )  # path to the figures directory
        self.logfile_path = Path(self.experiment_path, "logfiles")

    def _build_model(self):
        """
        Build the 'experiment' for the analysis

        A method to build the database_path in the hdf5 format. Within this method,
        several other are called to develop the database_path skeleton,
        get configurations, and process and store the configurations. The method is
        accompanied by a loading bar which should be customized to make it more
        interesting.
        """

        # Create new analysis directory and change into it
        try:
            self.experiment_path.mkdir()
            self.figures_path.mkdir()
            self.database_path.mkdir()
            self.logfile_path.mkdir()
        except FileExistsError:  # throw exception if the file exits
            return

        # self.save_class()  # save the class state.
        log.info(f"** An experiment has been added titled {self.name} **")

    @staticmethod
    def units_to_si(units_system):
        """
        Returns a dictionary with equivalences from the unit experiment given by a
        string to SI. Along with some constants in the unit experiment provided
        (boltzmann, or other conversions). Instead, the user may provide a dictionary.
        In that case, the dictionary will be used as the unit experiment.


        Parameters
        ----------
        units_system (str) -- current unit experiment
        dimension (str) -- dimension you would like to change

        Returns
        -------
        conv_factor (float) -- conversion factor to pass to SI
        """

        if isinstance(units_system, Units):
            return units_system
        elif isinstance(units_system, str):
            try:
                units = units_dict[units_system]
            except KeyError:
                raise KeyError(
                    f"The unit '{units_system}' is not implemented."
                    f" The available units are: {[x for x in units_dict]}"
                )
        else:
            raise ValueError(
                f"units has to be of type Units or str,"
                f" found {type(units_system)} instead"
            )
        return units

    def _load_or_build(self) -> bool:
        """
        Check if the experiment already exists and decide whether to load it or build a
        new one.
        """

        # Check if the experiment exists and load if it does.
        if Path(self.experiment_path).exists():
            log.debug("This experiment already exists! I'll load it up now.")
            # self.load_class()
            return True
        else:
            log.info("Creating a new experiment!")
            self._build_model()
            return False

    def perform_transformation(self, transformation_name, **kwargs):
        """
        Perform a transformation on the experiment.

        Parameters
        ----------
        transformation_name : str
                Name of the transformation to perform.
        **kwargs
                Other arguments associated with the transformation.

        Returns
        -------
        Update of the database_path.
        """

        try:
            transformation = transformations_dict[transformation_name]
        except KeyError:
            raise ValueError(
                f"{transformation_name} not found! \nAvailable transformations are:"
                f" {[key for key in transformations_dict]}"
            )

        transformation_run = transformation(self, **kwargs)
        transformation_run.run_transformation()  # perform the transformation

    def run_visualization(
        self,
        species: list = None,
        molecules: bool = False,
        unwrapped: bool = False,
        starting_configuration: int = None,
    ):
        """
        Perform a trajectory visualization.

        Parameters
        ----------
        species : list
                A list of species of molecules to study.
        molecules : bool
                If true, molecule groups will be used.
        unwrapped : bool
                If true, unwrapped coordinates will be visualized.
        starting_configuration : int
                Starting configuration for the visualizer.

        Returns
        -------
        Displays a visualization app.
        """
        if molecules:
            if species is None:
                species = list(self.species)
        if species is None:
            species = list(self.species)

        visualizer = SimulationVisualizer(
            self,
            species=species,
            molecules=molecules,
            unwrapped=unwrapped,
            number_of_configurations=self.number_of_configurations,
        )
        visualizer.run_app(starting_configuration=starting_configuration)

    # def map_elements(self, mapping: dict = None):
    #     """
    #     Map numerical keys to element names in the Experiment class and database_path.
    #
    #     Returns
    #     -------
    #     Updates the class
    #     """
    #
    #     if mapping is None:
    #         log.info("Must provide a mapping")
    #         return
    #
    #     # rename keys in species dictionary
    #     for item in mapping:
    #         self.species[mapping[item]] = self.species.pop(item)
    #
    #     # rename database_path groups
    #     db_object = Database(name=os.path.join(self.database_path,
    #     "database_path.hdf5"))
    #     db_object.change_key_names(mapping)
    #
    #     self.save_class()  # update the class state
    #
    #
    # def set_element(self, old_name, new_name):
    #     """
    #     Change the name of the element in the self.species dictionary
    #
    #     Parameters
    #     ----------
    #     old_name : str
    #             Name of the element you want to change
    #     new_name : str
    #             New name of the element
    #     """
    #     # Check if the new name is new
    #     if new_name != old_name:
    #         self.species[new_name] = self.species[old_name]  # update dict
    #         del self.species[old_name]  # remove old entry

    def set_charge(self, element: str, charge: float):
        """
        Set the charge/s of an element

        Parameters
        ----------
        element : str
                Name of the element whose charge you want to change
        charge : list
                New charge/s of the element
        """

        species = self.species
        species[element]["charge"] = [charge]  # update entry
        self.species = species

    def set_mass(self, element: str, mass: float):
        """
        Set the mass/es of an element

        Parameters
        ----------
        element : str
                Name of the element whose mass you want to change
        mass : list
                New mass/es of the element
        """
        species = self.species
        species[element]["mass"] = mass  # update the mass
        self.species = species

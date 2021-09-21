"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

The central experiment class fundamental to all analysis.

Summary
-------
The experiment class is the main class involved in characterizing and analyzing a simulation.
"""
import logging
from pathlib import Path
from mdsuite.calculators import RunComputation
from mdsuite.time_series import time_series_dict
from mdsuite.transformations.transformation_dict import transformations_dict
from mdsuite.utils.units import units_dict
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
            Path to where the tensor_values should be stored (best to have  drive capable of storing large files)
    temperature : float
            The temperature of the simulation that should be used in some analysis. Necessary as it cannot be easily
            read in from the simulation tensor_values.
    time_step : float
            Time step of the simulation e.g 0.002. Necessary as it cannot be easily read in from the trajectory.
    volume : float
            Volume of the simulation box
    species : dict
            A dictionary of the species in the experiment and their properties. Their properties includes
            index location in the trajectory file, mass of the species as taken from the PubChem
            database_path, and the charge taken from the same database_path. When using these properties, it is
            best that users confirm this information, with exception to the indices as they are read
            from the file and will be correct.
    number_of_atoms : int
            The total number of atoms in the simulation
   """

    def __init__(self, project, experiment_name, storage_path='./', time_step=None, temperature=None,
                 units: Union[str, dict] = None,
                 cluster_mode=False):
        """
        Initialise the experiment class.

        Attributes
        ----------
        experiment_name : str
                The name of the analysis being performed e.g. NaCl_1400K
        storage_path : str
                Path to where the tensor_values should be stored (best to have  drive capable of storing large files)
        temperature : float
                The temperature of the simulation that should be used in some analysis.
        time_step : float
                Time step of the simulation e.g 0.002. Necessary as it cannot be easily read in from the trajectory.
        units: Union[str, dict], default = "real"
            The units to be used in the experiment to convert to SI
        cluster_mode : bool
                If true, several parameters involved in plotting and parallelization will be adjusted so as to allow
                for optimal performance on a large computing cluster.
        """

        # Taken upon instantiation
        super().__init__(project=project, experiment_name=experiment_name)
        self.name = experiment_name  # Name of the experiment.
        self.storage_path = Path(project.storage_path, project.name).as_posix()
        self.cluster_mode = cluster_mode  # whether or not the script will run on a cluster

        # ExperimentDatabase stored properties:
        # ------- #
        # set default values
        if self.number_of_configurations is None:
            self.number_of_configurations = 0  # Number of configurations in the trajectory.
        # update database (None values are ignored)
        self.temperature = temperature  # Temperature of the experiment.
        self.time_step = time_step  # Timestep chosen for the simulation.

        # Available properties that aren't set on default
        self.number_of_atoms = None  # Number of atoms in the simulation.
        # ------- #

        # Added from trajectory file
        self.units = self.units_to_si(units)  # Units used during the simulation.
        self.species = None  # Species dictionary.
        self.molecules = {}  # molecules
        self.box_array = None  # Box vectors.
        self.dimensions = None  # Dimensionality of the experiment.

        self.sample_rate = None  # Rate at which configurations are dumped in the trajectory.
        self.batch_size = None  # Number of configurations in each batch.
        self.volume = None  # Volume of the experiment.
        self.properties = None  # Properties measured in the simulation.
        self.property_groups = None  # Names of the properties measured in the simulation

        # Internal File paths
        self.experiment_path: str
        self.database_path: str
        self.figures_path: str
        self.logfile_path: str
        self._create_internal_file_paths()  # fill the path attributes

        # self.radial_distribution_function_state = False  # Set true if this has been calculated
        self.kirkwood_buff_integral_state = False  # Set true if it has been calculated
        self.structure_factor_state = False

        # Memory properties
        self.memory_requirements = {}  # TODO I think this can be removed. - Not until all calcs are under tf Dataset

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

    @property
    def load(self) -> RunComputation:
        """Method to access the available calculators results

        Returns
        -------
        RunComputation:
            if called, return List[db.Computation]

        """
        return RunComputation(experiment=self, load_data=True)

    def __repr__(self):
        return f"exp_{self.name}"

    def _create_internal_file_paths(self):
        """
        Create or update internal file paths
        """
        self.experiment_path = Path(self.storage_path, self.name)  # path to the experiment files
        self.database_path = Path(self.experiment_path, 'databases')  # path to the databases
        self.figures_path = Path(self.experiment_path, 'figures')  # path to the figures directory
        self.logfile_path = Path(self.experiment_path, 'logfiles')

    def _build_model(self):
        """
        Build the 'experiment' for the analysis

        A method to build the database_path in the hdf5 format. Within this method, several other are called to develop
        the database_path skeleton, get configurations, and process and store the configurations. The method is
        accompanied by a loading bar which should be customized to make it more interesting.
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
        Returns a dictionary with equivalences from the unit experiment given by a string to SI.
        Along with some constants in the unit experiment provided (boltzmann, or other conversions).
        Instead, the user may provide a dictionary. In that case, the dictionary will be used as the unit experiment.


        Parameters
        ----------
        units_system (str) -- current unit experiment
        dimension (str) -- dimension you would like to change

        Returns
        -------
        conv_factor (float) -- conversion factor to pass to SI
        """

        if units_system is None:
            units_system = "real"  # set default here!

        if isinstance(units_system, dict):
            return units_system
        else:
            try:
                units = units_dict[units_system]()
                # executes the function to return the appropriate dictionary.
                # TODO Why is this a function?!
            except KeyError:
                raise KeyError(
                    f"The unit '{units_system}' is not implemented."
                    f" The available units are: {[x for x in units_dict]}"
                )
        return units

    #
    def _load_or_build(self) -> bool:
        """
        Check if the experiment already exists and decide whether to load it or build a new one.
        """

        # Check if the experiment exists and load if it does.
        if Path(self.experiment_path).exists():
            log.info("This experiment already exists! I'll load it up now.")
            # self.load_class()
            return True
        else:
            log.info("Creating a new experiment!")
            self._build_model()
            return False

    def save_class(self):
        log.warning("Using depreciated method `save_class`!")

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
                f'{transformation_name} not found! \n'
                f'Available transformations are: {[key for key in transformations_dict]}'
            )

        transformation_run = transformation(self, **kwargs)
        transformation_run.run_transformation()  # perform the transformation

    def run_visualization(
            self,
            species: list = None,
            molecules: bool = False,
            unwrapped: bool = False,
            starting_configuration: int = None):
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
            self, species=species, molecules=molecules, unwrapped=unwrapped,
            number_of_configurations=self.number_of_configurations)
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
    #     db_object = Database(name=os.path.join(self.database_path, "database_path.hdf5"))
    #     db_object.change_key_names(mapping)
    #
    #     self.save_class()  # update the class state
    #

    #
    # def print_class_attributes(self):
    #     """
    #     Print all attributes of the class
    #
    #     Returns
    #     -------
    #     attributes : list
    #             List of class attribute tuples of (key, value)
    #     """
    #
    #     attributes = []  # define empty array
    #     for item in vars(self).items():  # loop over class attributes
    #         attributes.append(item)  # append to the attributes array
    #     for tuple_attributes in attributes:  # Split the key and value terms
    #         log.info(f"{tuple_attributes[0]}: {tuple_attributes[1]}")
    #
    #     return attributes
    #
    #
    #
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
    #
    # def set_charge(self, element: str, charge: float):
    #     """
    #     Set the charge/s of an element
    #
    #     Parameters
    #     ----------
    #     element : str
    #             Name of the element whose charge you want to change
    #     charge : list
    #             New charge/s of the element
    #     """
    #     self.species[element]['charge'] = [charge]  # update entry
    #     self.save_class()
    #
    # def set_mass(self, element: str, mass: float):
    #     """
    #     Set the mass/es of an element
    #
    #     Parameters
    #     ----------
    #     element : str
    #             Name of the element whose mass you want to change
    #     mass : list
    #             New mass/es of the element
    #     """
    #     self.species[element]['mass'] = mass  # update the mass
    #
    #
    # @property
    # def results(self):
    #     """
    #     Property to get access to the results in a dictionary
    #
    #     Returns
    #     -------
    #     self._results: dict
    #         the actual dictionary with the results
    #     """
    #     return self._results
    #
    # @results.getter
    # def results(self):
    #     """
    #     Getter to retrieve the results from the YAML file in a dictionary
    #     :return: dict
    #
    #     Returns
    #     -------
    #     self._results: dict
    #         the actual dictionary with the results from the YAML file
    #     """
    #
    #     with open(os.path.join(self.database_path, 'system_properties.yaml')) as pfr:
    #         self._results = yaml.load(pfr, Loader=yaml.Loader)  # collect the data in the yaml file
    #
    #     return self._results
    #
    # @results.setter
    # def results(self, result_dict: dict):
    #     """
    #     Setter to dump the results to the YAML file
    #
    #     Parameters
    #     ----------
    #     result_dict: dict
    #         dictionary with the results. It will store them in the system_properties.yaml file.
    #
    #     """
    #
    #     with open(os.path.join(self.database_path, 'system_properties.yaml'), 'w') as pfw:
    #         yaml.dump(result_dict, pfw)
    #
    # def write_xyz(self, dump_property: str = "Positions", species: list = None, name: str = 'dump.xyz'):
    #     """
    #     Write an xyz file from a database dataset
    #     """
    #     if species is None:
    #         species = list(self.species)
    #
    #     database = Database(name=os.path.join(self.database_path, "database.hdf5"), architecture='simulation')
    #
    #     path_list = [join_path(s, dump_property) for s in species]
    #     if len(species) == 1:
    #         data_matrix = [database.load_data(path_list=path_list, select_slice=np.s_[:, 0:500])]
    #     else:
    #         data_matrix = database.load_data(path_list=path_list, select_slice=np.s_[:, 0:500])
    #     n_atoms = sum(len(self.species[s]['indices']) for s in species)
    #
    #     with open(f"{name}.xyz", 'w') as f:
    #         for i in tqdm(range(500), ncols=70):
    #             f.write(f"{n_atoms}\n")
    #             f.write("Generated by the mdsuite xyz writer\n")
    #             for j, atom_species in enumerate(species):
    #                 for atom in data_matrix[j].numpy():
    #                     f.write(
    #                         f"{atom_species:<2}    {atom[i][0]:>9.4f}    {atom[i][1]:>9.4f}    {atom[i][2]:>9.4f}\n")
    #
    # def export_data(self, group: str, key: str = None, sub_key: str = None):
    #     """
    #     Export data from the analysis database.
    #
    #     Parameters
    #     ----------
    #     group : str
    #             Group in the database from which data should be loaded
    #     key  : str
    #             Additional identifier.
    #     sub_key : str
    #             Additional identifier
    #     Returns
    #     -------
    #     saves a csv to the working directory.
    #     """
    #     database = Database(name=os.path.join(self.database_path, 'analysis_data.hdf5'), architecture='analysis')
    #     database.export_csv(group=group, key=key, sub_key=sub_key)
    #

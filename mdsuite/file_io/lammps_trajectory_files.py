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
"""
import tqdm

import mdsuite.file_io.tabular_text_files
import mdsuite.utils.meta_functions
import numpy as np
import copy
import typing

from mdsuite.database.simulation_database import (
    TrajectoryMetadata,
    PropertyInfo,
    SpeciesInfo,
)
from mdsuite.utils.meta_functions import sort_array_by_column

var_names = {
    "Positions": ["x", "y", "z"],
    "Scaled_Positions": ["xs", "ys", "zs"],
    "Unwrapped_Positions": ["xu", "yu", "zu"],
    "Scaled_Unwrapped_Positions": ["xsu", "ysu", "zsu"],
    "Velocities": ["vx", "vy", "vz"],
    "Forces": ["fx", "fy", "fz"],
    "Box_Images": ["ix", "iy", "iz"],
    "Dipole_Orientation_Magnitude": ["mux", "muy", "muz"],
    "Angular_Velocity_Spherical": ["omegax", "omegay", "omegaz"],
    "Angular_Velocity_Non_Spherical": ["angmomx", "angmomy", "angmomz"],
    "Torque": ["tqx", "tqy", "tqz"],
    "Charge": ["q"],
    "KE": ["c_KE"],
    "PE": ["c_PE"],
    "Stress": [
        "c_Stress[1]",
        "c_Stress[2]",
        "c_Stress[3]",
        "c_Stress[4]",
        "c_Stress[5]",
        "c_Stress[6]",
    ],
}


class LAMMPSTrajectoryFile(mdsuite.file_io.tabular_text_files.TabularTextFileProcessor):
    """
    Reader for LAMMPS files
    """

    def __init__(
        self,
        file_path: str,
        trajectory_is_sorted_by_ids=False,
        custom_data_map: dict = None,
    ):
        """

        Parameters
        ----------
        file_path:
            path to the LAMMPS trajectory file
        trajectory_is_sorted_by_ids:
            Flag to indicate if the particle order in the trajectory file is the same for each configuration
        custom_data_map
            If your file contains columns with data that is not part of the standard set of properties (see var_names),
            you can map the column names to the corresponding property.
            example: custom_data_map = {"Reduced_Momentum": ["rp_x", "rp_y", "rp_z"]}, if the file contains columns
            labelled as 'rp_{x,y,z}' for the three components of the reduced momentum vector
        """
        super(LAMMPSTrajectoryFile, self).__init__(
            file_path, custom_data_map=custom_data_map
        )
        self.n_header_lines = 9
        self.trajectory_is_sorted_by_ids = trajectory_is_sorted_by_ids

        # attributes that will be filled in by get_metadata() and are later used by get_configurations_generator()
        self._batch_size = None
        self._id_column_idx = None
        self._n_particles = None
        self._species_dict = None
        self._property_dict = None
        self._mdata = None

    def get_metadata(self) -> TrajectoryMetadata:
        """
        Gets the metadata for database creation.
        Also creates the lookup dictionaries on where to find the particles and properties in the file
        """
        with open(self.file_path, "r") as file:
            header = mdsuite.file_io.tabular_text_files.read_n_lines(
                file, self.n_header_lines, start_at=0
            )

            # extract data that can be directly read off the header
            self._n_particles = int(header[3].split()[0])
            header_boxl_lines = header[5:8]
            box_l = [
                float(line.split()[1]) - float(line.split()[0])
                for line in header_boxl_lines
            ]

            # extract properties from the column names
            header_property_names = header[8].split()[2:]
            self._id_column_idx = header_property_names.index("id")
            updated_var_names = copy.deepcopy(var_names)
            updated_var_names.update(self.custom_data_map)
            self._property_dict = (
                mdsuite.file_io.tabular_text_files.extract_properties_from_header(
                    header_property_names, updated_var_names
                )
            )

            # get number of configs from file length
            file.seek(0)
            num_lines = sum(1 for _ in file)
            n_configs_float = num_lines / (self._n_particles + self.n_header_lines)
            n_configs = int(round(n_configs_float))
            assert abs(n_configs_float - n_configs) < 1e-10

            # get information on which particles with which id belong to which species
            # by analysing the first configuration
            file.seek(0)
            self._species_dict = self._get_species_information(
                file, header_property_names
            )

            # extract sample step information from consecutive headers
            file.seek(0)
            sample_rate = self._get_sample_rate(file)

        # all species have the same properties
        properties_list = list()
        for key, val in self._property_dict.items():
            properties_list.append(PropertyInfo(name=key, n_dims=len(val)))

        species_list = list()
        for key, val in self._species_dict.items():
            species_list.append(
                SpeciesInfo(
                    name=key,
                    n_particles=len(val),
                    properties=properties_list,
                )
            )

        self._mdata = TrajectoryMetadata(
            n_configurations=n_configs,
            species_list=species_list,
            box_l=box_l,
            sample_rate=sample_rate,
        )

        self._batch_size = mdsuite.utils.meta_functions.optimize_batch_size(
            filepath=self.file_path, number_of_configurations=n_configs
        )

        return self._mdata

    def get_configurations_generator(
        self,
    ) -> typing.Iterator[mdsuite.database.simulation_database.TrajectoryChunkData]:
        n_configs = self._mdata.n_configurations
        n_batches, n_configs_remainder = divmod(int(n_configs), int(self._batch_size))

        sort_by_column_idx = (
            None if self.trajectory_is_sorted_by_ids else self._id_column_idx
        )

        with open(self.file_path, "r") as file:
            file.seek(0)
            for _ in tqdm.tqdm(range(n_batches)):
                yield mdsuite.file_io.tabular_text_files._read_process_n_configurations(
                    file,
                    self._batch_size,
                    self._mdata.species_list,
                    self._species_dict,
                    self._property_dict,
                    self._n_particles,
                    n_header_lines=self.n_header_lines,
                    sort_by_column_idx=sort_by_column_idx,
                )
            if n_configs_remainder > 0:
                yield mdsuite.file_io.tabular_text_files._read_process_n_configurations(
                    file,
                    n_configs_remainder,
                    self._mdata.species_list,
                    self._species_dict,
                    self._property_dict,
                    self._n_particles,
                    n_header_lines=self.n_header_lines,
                    sort_by_column_idx=sort_by_column_idx,
                )

    def _get_species_information(self, file, header_property_names: list):
        """
        Get the information which species are present and which particle ids/ lines in the file belong to them

        Parameters
        ----------
        file: file object
            the open file object to read from
        header_property_names: list
            list of the names of the columns in the file

        """
        header_id_index = header_property_names.index("id")
        #
        # Look for element keyword in trajectory.
        if "element" in header_property_names:
            header_species_index = header_property_names.index("element")
        # Look for type keyword if element is not present.
        elif "type" in header_property_names:
            header_species_index = header_property_names.index("type")
        # Raise an error if no identifying keywords are found.
        else:
            raise ValueError("Insufficient species or type identification available.")

        species_dict = dict()
        # skip the header
        mdsuite.file_io.tabular_text_files.skip_n_lines(file, self.n_header_lines)
        # read one configuration
        traj_data = np.stack(
            [np.array(list(file.readline().split())) for _ in range(self._n_particles)]
        )
        # sort by particle id
        if not self.trajectory_is_sorted_by_ids:
            traj_data = sort_array_by_column(traj_data, header_id_index)
        # iterate over the first configuration, whenever a new species (value at species_index) is encountered,
        # add an entry
        for i, line in enumerate(traj_data):
            species_name = line[header_species_index]
            if species_name not in species_dict.keys():
                species_dict[species_name] = []
            species_dict[species_name].append(i)

        return species_dict

    def _get_sample_rate(self, file) -> int:
        first_header = mdsuite.file_io.tabular_text_files.read_n_lines(
            file, self.n_header_lines, start_at=0
        )
        time_step_0 = int(first_header[1])  # Time in first configuration
        second_header = mdsuite.file_io.tabular_text_files.read_n_lines(
            file, self.n_header_lines, start_at=self.n_header_lines + self._n_particles
        )
        time__step_1 = int(second_header[1])  # Time in second configuration

        return time__step_1 - time_step_0

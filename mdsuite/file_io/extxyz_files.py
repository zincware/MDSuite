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
import typing
from typing import Union, List, Dict, Tuple
import numpy as np
import pathlib
import copy
import tqdm

import mdsuite.file_io.file_read
import mdsuite.database.simulation_database
from mdsuite.utils.meta_functions import get_dimensionality

log = logging.getLogger(__name__)

var_names = {
    "Positions": "pos",
    "Velocities": "vel",
    "Forces": "forces",
    "Stress": "stress",
    "PE": "energies",
    "Time": "time",
    "Lattice": "Lattice",
    "Momenta": "momenta",
}


class EXTXYZFile(mdsuite.file_io.file_read.FileProcessor):
    """
    Reader for extxyz files
    """

    def __init__(self, file_path: str, custom_data_map: dict = None):
        self.file_path = pathlib.Path(file_path).resolve()
        self.n_header_lines = 2
        if custom_data_map is None:
            custom_data_map = {}
        self.custom_data_map = custom_data_map

        self._n_particles = None
        self._mdata = None
        self._properties_dict = None
        self._species_dict = None
        self._batch_size = None

    def __str__(self):
        return str(self.file_path)

    def get_metadata(self):
        with open(self.file_path, "r") as file:
            # first header line: number of particles
            self._n_particles = int(file.readline())
            # second header line: properties
            header = file.readline()

            box_l = _get_box_l(header)

            var_names_updated = copy.deepcopy(var_names)
            var_names_updated.update(self.custom_data_map)
            species_idx, property_dict = _get_property_summary(
                header, var_names_updated
            )

            file.seek(0)
            self._species_dict = self._get_species_information(file, species_idx)

            file.seek(0)
            mdsuite.file_io.file_read.skip_n_lines(
                file, self._n_particles + self.n_header_lines + 1
            )
            header_1 = file.readline()
            sample_rate = int(round(_get_time(header_1) - _get_time(header)))

            # get number of configs from file length
            file.seek(0)
            num_lines = sum(1 for _ in file)
            n_configs_float = num_lines / (self._n_particles + self.n_header_lines)
            n_configs = int(round(n_configs_float))
            assert abs(n_configs_float - n_configs) < 1e-10

        # same properties for all species
        properties_list = []
        for prop_name, prop_col_idxs in self._properties_dict.items():
            properties_list.append(
                mdsuite.database.simulation_database.PropertyInfo(
                    name=prop_name, n_dims=len(prop_col_idxs)
                )
            )
        species_list = []
        for sp_name, sp_indices in self._species_dict.items():
            species_list.append(
                mdsuite.database.simulation_database.SpeciesInfo(
                    name=sp_name,
                    n_particles=len(sp_indices),
                    properties=properties_list,
                )
            )

        self._mdata = mdsuite.database.simulation_database.TrajectoryMetadata(
            n_configurations=n_configs,
            box_l=box_l,
            sample_rate=sample_rate,
            species_list=species_list,
        )
        self._batch_size = mdsuite.utils.meta_functions.optimize_batch_size(
            filepath=self.file_path, number_of_configurations=n_configs
        )

        return self._mdata

    def get_configurations_generator(
        self,
    ) -> typing.Iterator[mdsuite.file_io.file_read.TrajectoryChunkData]:
        n_configs = self._mdata.n_configurations
        n_batches, n_configs_remainder = divmod(int(n_configs), int(self._batch_size))

        with open(self.file_path, "r") as file:
            file.seek(0)
            for _ in tqdm.tqdm(range(n_batches)):
                yield self._read_process_n_configurations(file, self._batch_size)
            if n_configs_remainder > 0:
                yield self._read_process_n_configurations(file, n_configs_remainder)

    def _read_process_n_configurations(
        self, file, n_configs
    ) -> mdsuite.file_io.file_read.TrajectoryChunkData:
        """
        Read n_configs configurations and bring them to the structore needed for the yield of get_configurations_generator()
        Parameters
        ----------
        file
            The open trajectory file. Note: Calling this function will advance the reading point in the file
        n_configs

        Returns
        -------

        """
        chunk = mdsuite.file_io.file_read.TrajectoryChunkData(
            self._mdata.species_list, n_configs
        )

        for config_idx in range(n_configs):
            # skip the header
            mdsuite.file_io.file_read.skip_n_lines(file, self.n_header_lines)
            # read one config
            traj_data = np.stack(
                [
                    np.array(list(file.readline().split()))
                    for _ in range(self._n_particles)
                ]
            )

            # slice by species
            for sp_info in self._mdata.species_list:
                idxs = self._species_dict[sp_info.name]["line_idxs"]
                sp_data = traj_data[idxs, :]
                # slice by property
                for prop_info in sp_info.properties:
                    prop_column_idxs = self._properties_dict[prop_info.name]
                    write_data = sp_data[:, prop_column_idxs]
                    # add 'time' axis. we only have one configuration to write
                    write_data = write_data[np.newaxis, :, :]
                    chunk.add_data(write_data, config_idx, sp_info.name, prop_info.name)

        return chunk

    def _get_species_information(self, file, species_idx):
        """
        Get the initial species information

        Parameters
        ----------

        """
        mdsuite.file_io.file_read.skip_n_lines(file, self.n_header_lines)
        # read one configuration
        traj_data = np.stack(
            [np.array(list(file.readline().split())) for _ in range(self._n_particles)]
        )

        # Loop over atoms in first configuration.
        species_dict = {}
        for i, line in enumerate(traj_data):
            sp_name = line[species_idx]
            if sp_name not in list(species_dict.keys()):
                species_dict[sp_name]["indices"] = []

            species_dict[sp_name]["indices"].append(i)

        return species_dict


def _get_box_l(header: str) -> list:
    data = copy.deepcopy(header).split()
    lattice = None
    start = None
    stop = None
    for idx, item in enumerate(data):
        if var_names["Lattice"] in item:
            start = idx
            break

    if start is not None:
        for idx, item in enumerate(data[start:]):
            if item[-1] == '"':
                stop = idx
                break
    else:
        raise RuntimeError("Could not find lattice size in file header")

    if stop is not None:
        lattice = data[start : stop + 1]
        lattice[0] = lattice[0].split("=")[1].replace('"', "")
        lattice[-1] = lattice[-1].replace('"', "")
        lattice = np.array(lattice).astype(float)

    return [lattice[0], lattice[4], lattice[8]]


def _get_time(header: str):
    data = copy.deepcopy(header).split()
    time = None
    for item in data:
        if var_names["Time"] in item:
            try:
                time = float(item.split("=")[-1])
            except ValueError:
                time = float(item.split("=")[-1].split(",")[0])
    return time


def _get_property_summary(
    header: str, var_names: dict
) -> Tuple[int, Dict[str, List[int]]]:
    """
    Get the property summary from the header data.

    Parameters
    ----------
    header:
        header to analyse
    var_names:
        dict of translations from MDsuite property names to extxyz property names
    Returns
    -------
    property_summary : dict
            A dictionary of properties and their location in the data file.
    """
    data = copy.deepcopy(header).split()
    properties_string = None
    for item in data:
        if "Properties" in item:
            properties_string = item
    if properties_string is None:
        raise RuntimeError("Could not find properties in header")

    properties_list = (
        properties_string.split("=")[1].replace(":S", "").replace(":R", "").split(":")
    )
    property_summary = {}
    species_index = None
    index = 0
    var_names_values = list(var_names.values())
    var_names_keys = list(var_names.keys())
    for idx, item in enumerate(properties_list):
        if item == "species":
            species_index = index
            index += 1
        if item in var_names_values:
            key = var_names_keys[var_names_values.index(item)]
            length = int(properties_list[idx + 1])
            property_summary[key] = [index + i for i in range(length)]
            index += length

    if species_index is None:
        raise RuntimeError("could not find species in header")

    return species_index, property_summary

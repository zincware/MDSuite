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
import pathlib

import numpy as np
import typing
import tqdm
import mdsuite.file_io.file_read
import mdsuite.database.simulation_database
from mdsuite.utils.meta_functions import optimize_batch_size, join_path
import copy

var_names = {
    "Temperature": ["temp"],
    "Time": ["time"],
    "Thermal_Flux": ["c_flux_thermal[1]", "c_flux_thermal[2]", "c_flux_thermal[3]"],
    "Stress_visc": ["pxy", "pxz", "pyz"],
}


class LAMMPSFluxFile(mdsuite.file_io.file_read.FileProcessor):
    def __init__(self, file_path: str, sample_rate: int, box_l: list, n_header_lines: int = 2,
                 custom_data_map: dict = None):
        """
        Initialize the lammps flux reader. Since the flux file does not have a fixed expected content,
        you need to provide the necessary metadata (sample_rate, box_l) here manually
        Parameters
        ----------
        file_path
            Location of the file
        sample_rate
            Number of time steps between successive samples
        box_l
            Array of box lengths
        n_header_lines
            Number of header lines on the top of the file
            first (n_header_lines-1) lines will be skipped, line n_header_lines must contain the column names
        custom_data_map
            Dictionary connecting the name in the mdsuite database to the name of the corresponding columns
            example: {"Thermal_Flux": ["c_flux_thermal[1]", "c_flux_thermal[2]", "c_flux_thermal[3]"]}
        """
        self.file_path = pathlib.Path(file_path).resolve()
        self.sample_rate = sample_rate
        self.box_l = box_l

        self.n_header_lines = n_header_lines
        if custom_data_map is None:
            custom_data_map = {}
        self.custom_data_map = custom_data_map

        self._properties_dict = None
        self._batch_size = None
        self._mdata = None

    def __str__(self):
        return str(self.file_path)

    def get_metadata(self):

        with open(self.file_path, 'r') as file:
            num_lines = sum(1 for _ in file)
            n_steps = num_lines - self.n_header_lines

            file.seek(0)
            headers = mdsuite.file_io.file_read.read_n_lines(file, self.n_header_lines)
            column_header = headers[-1]
            updated_column_names = copy.deepcopy(var_names)
            updated_column_names.update(self.custom_data_map)
            self._properties_dict = mdsuite.file_io.file_read.extract_properties_from_header(column_header.split(),
                                                                                             updated_column_names)

        properties_list = []
        for prop_name, prop_idxs in self._properties_dict.items():
            properties_list.append(mdsuite.database.simulation_database.PropertyInfo(name=prop_name,
                                                                                     n_dims=len(prop_idxs)))
        species_list = [mdsuite.database.simulation_database.SpeciesInfo(name='Observables',
                                                                         n_particles=1,
                                                                         properties=properties_list)]
        self._mdata = mdsuite.database.simulation_database.TrajectoryMetadata(n_configurations=n_steps,
                                                                              species_list=species_list,
                                                                              box_l=self.box_l)

        self._batch_size = mdsuite.utils.meta_functions.optimize_batch_size(
            filepath=self.file_path, number_of_configurations=n_steps
        )

        return self._mdata

    def get_configurations_generator(
            self,
    ) -> typing.Iterator[mdsuite.file_io.file_read.TrajectoryChunkData]:
        n_configs = self._mdata.n_configurations
        n_batches, n_configs_remainder = divmod(int(n_configs), int(self._batch_size))

        with open(self.file_path, "r") as file:
            file.seek(0)
            # skip the header
            mdsuite.file_io.file_read.skip_n_lines(file, self.n_header_lines)
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

        traj_data = np.stack(
            [
                np.array(list(file.readline().split()))
                for _ in range(n_configs)
            ]
        )

        # there is only one species, containing the observable properties
        sp_name = self._mdata.species_list[0]
        properties = self._mdata.species_list[sp_name].properties
        for prop_info in properties:
            prop_column_idxs = self._properties_dict[prop_info.name]
            write_data = traj_data[:, prop_column_idxs]
            # add 'n_particles' axis. we only have one 'particle'
            write_data = write_data[:, np.newaxis, :]
            chunk.add_data(write_data, 0, sp_name, prop_info.name)

        return chunk


import dataclasses
import typing
import pathlib
import numpy as np
import mdsuite.file_io.file_read
import mdsuite.database.simulation_database
import mdsuite.utils.meta_functions
import abc
import tqdm


@dataclasses.dataclass
class TabularTextFileReaderMData:
    """
    Class to hold the data that needs to be extracted from tabular text files before reading them
    """

    n_configs: int
    species_name_to_line_idx_dict: typing.Dict[str, list]
    n_partilces: int
    property_to_column_idx_dict: typing.Dict[str, list]
    n_header_lines: int
    header_lines_for_each_config: bool = False
    sort_by_column_idx: int = None


class TabularTextFileProcessor(mdsuite.file_io.file_read.FileProcessor):
    """
    Parent class for all file readers that are based on tabular text data (e.g. lammps, extxyz,...)
    """

    def __init__(
        self,
        file_path: typing.Union[str, pathlib.Path],
        file_format_column_names: typing.Dict[
            mdsuite.database.simulation_database.PropertyInfo, list
        ] = None,
        custom_column_names: typing.Dict[str, typing.Any] = None,
    ):
        """
        Init, also handles the combination of file_format_column_names and custom_column_names.
        The result, self._column_name_dict is supposed to be used by child functions to create their TabularTextFileReaderData
        Parameters
        ----------
        file_path:
            Path to the tabular text file.
        file_format_column_names
            Dict connecting mdsuite properties (as defined in mdsuite.database.simulation_data_class.mdsuite_properties)
            the columns of the file format. Constant to be provided by the child classes.
            Example: {mdsuite_properties.positions: ["x", "y", "z"]}
        custom_column_names:
            Dict connecting user-defined properties the column names. To be provided by the user.
            Example: {'MyMagicProperty':['MMP1', 'MMP2']}
        """
        self.file_path = pathlib.Path(file_path).resolve()

        if file_format_column_names is None:
            file_format_column_names = {}
        str_file_format_column_names = {
            prop.name: val for prop, val in file_format_column_names.items()
        }

        if custom_column_names is None:
            custom_column_names = {}
        str_file_format_column_names.update(custom_column_names)
        self._column_name_dict = str_file_format_column_names

        self._tabular_text_reader_mdata: TabularTextFileReaderMData = None

    @abc.abstractmethod
    def _get_tabular_text_reader_mdata(self) -> TabularTextFileReaderMData:
        """
        Child classes of TabularTextFileProcessor must implement this function, so its output can be used in get_configurations_generator.
        See TabularTextFileReaderData for the data that needs to be provided
        """
        raise NotImplementedError("Tabular text files must implement this function")

    @property
    def tabular_text_reader_data(self) -> TabularTextFileReaderMData:
        if self._tabular_text_reader_mdata is None:
            self._tabular_text_reader_mdata = self._get_tabular_text_reader_mdata()
        return self._tabular_text_reader_mdata

    def __str__(self):
        return str(self.file_path)

    def get_configurations_generator(
        self,
    ) -> typing.Iterator[mdsuite.database.simulation_database.TrajectoryChunkData]:
        """
        TabularTextFiles implement the parent function,
        but requires its children to provide the necessary information about the table contents, see self._get_tabular_text_reader_data
        Returns
        -------

        """
        n_configs = self.tabular_text_reader_data.n_configs

        batch_size = mdsuite.utils.meta_functions.optimize_batch_size(
            filepath=self.file_path, number_of_configurations=n_configs
        )
        n_batches, n_configs_remainder = divmod(int(n_configs), int(batch_size))

        with open(self.file_path, "r") as file:
            file.seek(0)
            # skip header either once in the beginning or for each config
            if self.tabular_text_reader_data.header_lines_for_each_config:
                n_header_lines_in_config = self.tabular_text_reader_data.n_header_lines
            else:
                skip_n_lines(file, self.tabular_text_reader_data.n_header_lines)
                n_header_lines_in_config = 0

            for _ in tqdm.tqdm(range(n_batches)):
                yield mdsuite.file_io.tabular_text_files.read_process_n_configurations(
                    file,
                    batch_size,
                    self.metadata.species_list,
                    self.tabular_text_reader_data.species_name_to_line_idx_dict,
                    self.tabular_text_reader_data.property_to_column_idx_dict,
                    self.tabular_text_reader_data.n_partilces,
                    n_header_lines=n_header_lines_in_config,
                    sort_by_column_idx=self.tabular_text_reader_data.sort_by_column_idx,
                )
            if n_configs_remainder > 0:
                yield mdsuite.file_io.tabular_text_files.read_process_n_configurations(
                    file,
                    n_configs_remainder,
                    self.metadata.species_list,
                    self.tabular_text_reader_data.species_name_to_line_idx_dict,
                    self.tabular_text_reader_data.property_to_column_idx_dict,
                    self.tabular_text_reader_data.n_partilces,
                    n_header_lines=n_header_lines_in_config,
                    sort_by_column_idx=self.tabular_text_reader_data.sort_by_column_idx,
                )


def read_n_lines(file, n_lines: int, start_at: int = None) -> list:
    """
    Get n_lines lines, starting at line number start_at.
    If start_at is None, read from the current file state
    Returns
    -------
    A list of strings, one string for each line
    """
    if start_at is not None:
        file.seek(0)
        skip_n_lines(file, start_at)
    return [next(file) for _ in range(n_lines)]


def skip_n_lines(file, n_lines: int) -> None:
    """
    skip n_lines in file
    Parameters
    ----------
    file: the file where we skip lines
    n_lines: the number of lines to skip

    Returns
    -------
        Nothing
    """
    for _ in range(n_lines):
        next(file)


def read_process_n_configurations(
    file,
    n_configs: int,
    species_list: typing.List[mdsuite.database.simulation_database.SpeciesInfo],
    species_to_line_idx_dict: typing.Dict[str, list],
    property_to_column_idx_dict: typing.Dict[str, list],
    n_lines_per_config: int,
    n_header_lines: int = 0,
    sort_by_column_idx: int = None,
) -> mdsuite.database.simulation_database.TrajectoryChunkData:
    """
    Read n configurations and package them into a trajectory chunk of the right format.
    Parameters
    ----------
    file:
        A file opened at the start of a configuration
    n_configs:
        Number of configs to process
    species_list: List[mdsuite.database.simulation_database.SpeciesInfo]
        Species and property information as required by mdsuite.database.simulation_database.TrajectoryMetaData
    species_to_line_idx_dict:
        A dict that links the species name to the line idxs at which the particles can be found within a configuration.
        Example {"Na":[0,2,4], "Cl":[2,3,5]}
    property_to_column_idx_dict
        A dict that links the property name to the column idxs at which the property is listed.
        Usually the output of mdsuite.file_io.tabular_text_files.extract_properties_from_header
    n_lines_per_config
        Number of lines per config (= number of particles)
    n_header_lines:
        Number of header lines PER CONFIG
    sort_by_column_idx:
        if None (default): no effect
        if int: sort the lines in the config by the column with this index
        (e.g., use to sort by particle id in unsorted config output)
    Returns
    -------
        The chunk for your reader output
    """
    chunk = mdsuite.database.simulation_database.TrajectoryChunkData(
        species_list, n_configs
    )

    for config_idx in range(n_configs):
        # skip the header
        mdsuite.file_io.tabular_text_files.skip_n_lines(file, n_header_lines)
        # read one config
        traj_data = np.stack(
            [(list(file.readline().split())) for _ in range(n_lines_per_config)]
        )
        # sort by id
        if sort_by_column_idx is not None:
            traj_data = mdsuite.utils.meta_functions.sort_array_by_column(
                traj_data, sort_by_column_idx
            )

        # slice by species
        for sp_info in species_list:
            idxs = species_to_line_idx_dict[sp_info.name]
            sp_data = traj_data[idxs, :]
            # slice by property
            for prop_info in sp_info.properties:
                prop_column_idxs = property_to_column_idx_dict[prop_info.name]
                write_data = sp_data[:, prop_column_idxs]
                # add 'time' axis. we only have one configuration to write
                write_data = write_data[np.newaxis, :, :]
                chunk.add_data(write_data, config_idx, sp_info.name, prop_info.name)

    return chunk


def get_species_list_from_tabular_text_reader_data(
    tabular_text_reader_data: TabularTextFileReaderMData,
) -> typing.List[mdsuite.database.simulation_database.SpeciesInfo]:
    """
    Use the data collected in TabularTextFileProcessor._get_tabular_text_reader_data() to get the values necessary for
    TabularTextFileProcessor.metadata
    """
    # all species have the same properties
    properties_list = list()
    for (
        key,
        val,
    ) in tabular_text_reader_data.property_to_column_idx_dict.items():
        properties_list.append(
            mdsuite.database.simulation_database.PropertyInfo(name=key, n_dims=len(val))
        )

    species_list = list()
    for key, val in tabular_text_reader_data.species_name_to_line_idx_dict.items():
        species_list.append(
            mdsuite.database.simulation_database.SpeciesInfo(
                name=key,
                n_particles=len(val),
                properties=properties_list,
            )
        )

    return species_list

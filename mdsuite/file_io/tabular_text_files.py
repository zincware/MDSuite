import typing
import pathlib
import numpy as np
import mdsuite.file_io.file_read
import mdsuite.database.simulation_database
import mdsuite.utils.meta_functions


class TabularTextFileProcessor(mdsuite.file_io.file_read.FileProcessor):
    def __init__(
        self,
        file_path: typing.Union[str, pathlib.Path],
        file_format_column_names: typing.Dict[
            mdsuite.database.simulation_database.PropertyInfo, list
        ] = None,
        custom_column_names: typing.Dict[str, list] = None,
    ):
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

    def __str__(self):
        return str(self.file_path)


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


def extract_properties_from_header(
    header_property_names: list, database_correspondence_dict: dict
) -> dict:
    """
    Takes the property names from a file header, sees if there is a corresponding mdsuite property
    in database_correspondence_dict.
    Returns a dict that links the mdsuite property names to the column indices at which they can be found in the file.

    Parameters
    ----------
    header_property_names
        The names of the columns in the data file
    database_correspondence_dict
        The translation between mdsuite properties and the column names of the respective file format.
        lammps example:
        {"Positions": ["x", "y", "z"],
         "Unwrapped_Positions": ["xu", "yu", "zu"],

    Returns
    -------
    trajectory_properties : dict
        A dict of the form {'MDSuite_Property_1': [column_indices], 'MDSuite_Property_2': ...}
        Example {'Unwrapped_Positions': [2,3,4], 'Velocities': [5,6,7]}
    """

    column_dict_properties = {
        variable: idx for idx, variable in enumerate(header_property_names)
    }
    # for each property label (position, velocity,etc) in the lammps definition
    for property_label, property_names in database_correspondence_dict.items():
        # for each coordinate for a given property label (position: x, y, z),
        # get idx and the name
        for idx, property_name in enumerate(property_names):
            if (
                property_name in column_dict_properties.keys()
            ):  # if this name (x) is in the input file properties
                # we change the lammps_properties_dict replacing the string of the
                # property name by the column name
                database_correspondence_dict[property_label][
                    idx
                ] = column_dict_properties[property_name]

    # trajectory_properties only need the labels with the integer columns, then we
    # only copy those
    trajectory_properties = {}
    for property_label, properties_columns in database_correspondence_dict.items():
        if all(
            [isinstance(property_column, int) for property_column in properties_columns]
        ):
            trajectory_properties[property_label] = properties_columns

    return trajectory_properties


def read_process_n_configurations(
    file,
    n_configs: int,
    species_list: typing.List[mdsuite.database.simulation_database.SpeciesInfo],
    species_to_line_idx_dict: dict,
    property_to_column_idx_dict: dict,
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
        (use to sort by particle id in unsorted config output)

    Returns
    -------
        The chunk for you reader output
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

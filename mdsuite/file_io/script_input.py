import mdsuite.file_io.file_read
import typing


class SciptInput(mdsuite.file_io.file_read.FileProcessor):
    """
    For testing purposes. Does not actually process files, instead uses data given on instantiation
    """

    def __init__(self,
                 data: mdsuite.file_io.file_read.TrajectoryChunkData,
                 metadata: mdsuite.file_io.file_read.TrajectoryMetadata):
        self.data = data
        self.metadata = metadata

        mdsuite.file_io.file_read.assert_species_list_consistent(data.species_list, metadata.species_list)
        if self.data.chunk_size != self.metadata.n_configurations:
            raise ValueError('Data must be provided in one chunk')

    def get_metadata(self) -> mdsuite.file_io.file_read.TrajectoryMetadata:
        return self.metadata

    def get_configurations_generator(self) -> typing.Iterator[mdsuite.file_io.file_read.TrajectoryChunkData]:
        yield self.data

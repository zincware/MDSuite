"""
Class for memory management in MDSuite operations.
"""

from mdsuite.utils.meta_functions import get_machine_properties
from mdsuite.database.database import Database
import numpy as np
import sys


class MemoryManager:
    """
    A class to manage memory during MDSuite operations

    In most MDSuite operations the memory requirements of a complete procedure will be greater than what is available to
    the average computer. Therefore, some degree of memory management is required for smooth running. This class should
    be instantiated during any operation so it can update the operation on what the current memory capabilities are.

    The class can work with several data-sets in the case an analysis requires this much data.
    """

    def __init__(self, data_path: list = None, database: Database = None, scaling_factor: int = 1,
                 parallel: bool = False, memory_fraction: float = 0.5):
        """
        Constructor for the memory management class
        """
        self.data_path = data_path
        self.scaling_factor = scaling_factor
        self.parallel = parallel
        self.database = database
        self.memory_fraction = memory_fraction

        self.machine_properties = get_machine_properties()

        self.batch_size = None
        self.n_batches = None

    def get_batch_size(self) -> tuple:
        """
        Calculate the batch size of an operation.

        This method takes the data requirements of an operation and returns how big each batch of data should be for
        such an operation.

        Parameters
        ----------
        data_range : int
                Data range of the analysis taking place
        properties : list
                Properties to be loaded for the analysis
        Returns
        -------
        batch_size : int
                number of elements that should be loaded at one time
        remainder : int
                number of elements that will be left unloaded after a loop over all batches. This amount can then be
                loaded to collect unused data.
        """
        if self.data_path is []:
            print("No data has been requested.")
            sys.exit(1)
        per_configuration_memory: float = 0
        for item in self.data_path:
            n_rows, n_columns, n_bytes = self.database.get_data_size(item)
            per_configuration_memory += n_bytes / n_columns
        maximum_loaded_configurations = int(np.clip((self.memory_fraction*self.machine_properties['memory']) /
                                                    self.scaling_factor*per_configuration_memory, 1, n_columns))
        batch_size = self._get_optimal_batch_size(maximum_loaded_configurations)
        number_of_batches = int(n_columns / maximum_loaded_configurations)

        self.batch_size = batch_size
        self.n_batches = number_of_batches

        return batch_size, number_of_batches

    def _get_optimal_batch_size(self, naive_size):
        """
        Use the open/close and read speeds of the hdf5 database as well as the operation being performed to get an
        optimal batch size.

        Parameters
        ----------
        naive_size : int
                Naive batch size to be optimized

        Returns
        -------
        batch_size : int
                An optimized batch size
        """
        db_io_time = self.database.get_load_time()

        return naive_size

    def get_data_range_partitions(self, data_range: int, correlation_time: int = 1) -> int:
        """
        Get the data range partition quantity.

        Not only does data need to be batched, it then needs to be looped over in order to calculate some property. This
        method will return the number of loops possible given the data range and the correlation time

        Parameters
        ----------
        data_range : int
                Data range to be used in the analysis.
        correlation_time : int
                Correlation time to be considered when looping over the data
        Returns
        -------
        data_range_partitions : int
                Number of time the batch can be looped over with given data_range and correlation time.
        """
        final_window = self.batch_size - data_range
        if final_window < 0:
            print("Sub-batching is required for this system, or a smaller data_range can be used")
            sys.exit(1)
        else:
            return int(np.clip(final_window / correlation_time, 1))

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

    The class can work with several tensor_values-sets in the case an analysis requires this much tensor_values.
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
        self.remainder = None

    def get_batch_size(self, system: bool = False) -> tuple:
        """
        Calculate the batch size of an operation.

        This method takes the tensor_values requirements of an operation and returns how big each batch of tensor_values should be for
        such an operation.

        Parameters
        ----------
        system : bool
                Tell the database what kind of tensor_values it is looking at, atomistic, or system wide.
        Returns
        -------
        batch_size : int
                number of elements that should be loaded at one time
        remainder : int
                number of elements that will be left unloaded after a loop over all batches. This amount can then be
                loaded to collect unused tensor_values.
        """
        if self.data_path is []:
            print("No tensor_values has been requested.")
            sys.exit(1)
        per_configuration_memory: float = 0
        for item in self.data_path:
            n_rows, n_columns, n_bytes = self.database.get_data_size(item, system=system)
            per_configuration_memory += (n_bytes / n_columns)*self.scaling_factor
        maximum_loaded_configurations = int(np.clip((self.memory_fraction*self.machine_properties['memory']) /
                                                    per_configuration_memory, 1, n_columns))
        batch_size = self._get_optimal_batch_size(maximum_loaded_configurations)
        # batch_size = 100
        number_of_batches = int(n_columns / batch_size)
        remainder = int(n_columns % batch_size)

        self.batch_size = batch_size
        self.n_batches = number_of_batches
        self.remainder = remainder

        return batch_size, number_of_batches, remainder

    def hdf5_load_time(self, N):
        """
        Describes the load time of a hdf5 database i.e. O(log N)

        Parameters
        ----------
        N : int
                Amount of data to be loaded

        Returns
        -------

        """
        yield np.log(N)

    def _get_optimal_batch_size(self, naive_size):
        """
        Use the open/close and read speeds of the hdf5 database_path as well as the operation being performed to get an
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

    def get_ensemble_loop(self, data_range: int, correlation_time: int = 1) -> int:
        """
        Get the tensor_values range partition quantity.

        Not only does tensor_values need to be batched, it then needs to be looped over in order to calculate some property. This
        method will return the number of loops possible given the tensor_values range and the correlation time

        Parameters
        ----------
        data_range : int
                Data range to be used in the analysis.
        correlation_time : int
                Correlation time to be considered when looping over the tensor_values
        Returns
        -------
        data_range_partitions : int
                Number of time the batch can be looped over with given data_range and correlation time.
        """
        final_window = self.batch_size - data_range
        if final_window < 0:
            print("Sub-batching is required for this experiment, or a smaller data_range can be used")
            sys.exit(1)
        else:
            return int(np.clip(final_window / correlation_time, 1, None))

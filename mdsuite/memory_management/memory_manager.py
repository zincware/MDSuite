"""
Class for memory management in MDSuite operations.
"""

from mdsuite.utils.meta_functions import get_machine_properties
from mdsuite.database.simulation_database import Database
from mdsuite.utils.scale_functions import *
import numpy as np
import sys
import tensorflow as tf


class MemoryManager:
    """
    A class to manage memory during MDSuite operations

    In most MDSuite operations the memory requirements of a complete procedure will be greater than what is available to
    the average computer. Therefore, some degree of memory management is required for smooth running. This class should
    be instantiated during any operation so it can update the operation on what the current memory capabilities are.

    The class can work with several tensor_values-sets in the case an analysis requires this much tensor_values.
    """

    def __init__(self, data_path: list = None, database: Database = None, parallel: bool = False,
                 memory_fraction: float = 0.2, scale_function: dict = None, gpu: bool = False):
        """
        Constructor for the memory management class
        """
        if scale_function is None:
            scale_function = {'linear': {'scale_factor': 10}}
        self.data_path = data_path
        self.parallel = parallel
        self.database = database
        self.memory_fraction = memory_fraction

        self.machine_properties = get_machine_properties()
        if gpu:
            memory = 0
            for item in self.machine_properties['gpu']:
                if self.machine_properties['gpu'][item]['memory'] > memory:
                    memory = self.machine_properties['gpu'][item]['memory']

            self.machine_properties['memory'] = memory*1e6
            tf.device('gpu')
        else:
            tf.device('cpu')

        self.batch_size = None
        self.n_batches = None
        self.remainder = None

        self.scale_function, self.scale_function_parameters = self._select_scale_function(scale_function)

    @staticmethod
    def _select_scale_function(input_dict: dict):
        """
        select a scale function

        Parameters
        ----------
        input_dict : dict
                Input dict which selects the correct function.

        Returns
        -------
        Updates the class state
        """

        def _string_to_function(argument: str):
            """
            Convert a string to a function choice.

            Parameters
            ----------
            argument : str
                    String to use in the matching.
            Returns
            -------
            function : Callable
            """
            switcher = {
                'linear': linear_scale_function,
                'log-linear': linearithmic_scale_function,
                'quadratic': quadratic_scale_function,
                'polynomial': polynomial_scale_function
            }
            return switcher.get(argument, lambda: "Invalid choice")

        scale_function = _string_to_function(list(input_dict.keys())[0])
        scale_function_parameters = input_dict[list(input_dict.keys())[0]]

        return scale_function, scale_function_parameters

    def get_batch_size(self, system: bool = False) -> tuple:
        """
        Calculate the batch size of an operation.

        This method takes the tensor_values requirements of an operation and returns how big each batch of tensor_values
        should be for such an operation.


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
            print("No tensor_values have been requested.")
            sys.exit(1)
        per_configuration_memory: float = 0
        for item in self.data_path:
            n_rows, n_columns, n_bytes = self.database.get_data_size(item, system=system)
            per_configuration_memory += (n_bytes / n_columns)
        per_configuration_memory = self.scale_function(per_configuration_memory, **self.scale_function_parameters)
        maximum_loaded_configurations = int(np.clip((self.memory_fraction * self.machine_properties['memory']) /
                                                    per_configuration_memory, 1, n_columns))
        batch_size = self._get_optimal_batch_size(maximum_loaded_configurations)
        batch_size = 100 # forces batch size
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

        Not only does tensor_values need to be batched, it then needs to be looped over in order to calculate some
        property. This method will return the number of loops possible given the tensor_values range and the
        correlation time

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

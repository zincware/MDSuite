"""
Class for memory management in MDSuite operations.
"""

from mdsuite.utils.meta_functions import get_machine_properties


class MemoryManager:
    """
    A class to manage memory during MDSuite operations

    In most MDSuite operations the memory requirements of a complete procedure will be greater than what is available to
    the average computer. Therefore, some degree of memory management is required for smooth running. This class should
    be instantiated during any operation so it can update the operation on what the current memory capabilities are.

    The class can work with several data-sets in the case an analysis requires this much data.
    """

    def __init__(self):
        """
        Constructor for the memory management class
        """
        self.

        self.machine_properties = get_machine_properties()

    def get_batch_size(self, data_range: int, properties: list) -> tuple:
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
        pass

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
        pass


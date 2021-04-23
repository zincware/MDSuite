Frequently Asked Questions -- Developer Edition
===============================================
Here we will post up some of the frequently asked questions about code development in the hopes that it might answer
some questions of our developers without the need for further questioning.

* How do I manually set the batch size?
    - Great question! It is sometimes helpful to force a batch loop on data that is quite small as you want to check
      that everything is working with the batching. This can be done quite simply in the memory manager module in the
      get_batch_size method. In this method you will see the following lines:

        .. code-block:: python

            per_configuration_memory = self.scale_function(per_configuration_memory, **self.scale_function_parameters)
            maximum_loaded_configurations = int(np.clip((self.memory_fraction * self.machine_properties['memory']) /
                                                        per_configuration_memory, 1, n_columns))
            batch_size = self._get_optimal_batch_size(maximum_loaded_configurations)
            number_of_batches = int(n_columns / batch_size)
            remainder = int(n_columns % batch_size)
            self.batch_size = batch_size
            self.n_batches = number_of_batches
            self.remainder = remainder

        When you want to set the number of batches, simply set the variable batch size underneath the line

            batch_size = self._get_optimal_batch_size(maximum_loaded_configurations)

        This will set the batch size correctly and update the number of batches attribute for you.
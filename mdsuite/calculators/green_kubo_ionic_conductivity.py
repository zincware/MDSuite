"""
Class for the calculation of the Green-Kubo ionic conductivity.

Summary
This module contains the code for the Green-Kubo ionic conductivity class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.green_kubo_ionic_conductivity method.
The methods in class can then be called by the Experiment.green_kubo_ionic_conductivity method and all necessary
calculations performed.
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
import tensorflow as tf

# Import user packages
from tqdm import tqdm
import h5py as hf

# Import MDSuite modules
from mdsuite.utils.meta_functions import join_path
from mdsuite.utils.units import boltzmann_constant, elementary_charge
from mdsuite.database.database import Database
from mdsuite.calculators.calculator import Calculator

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboIonicConductivity(Calculator):
    """ Class for the Green-Kubo ionic conductivity implementation

    Attributes
    ----------
    obj :  object
            Experiment class to call from
    plot : bool
            if true, plot the data
    data_range :
            Number of configurations to use in each ensemble
    save :
            If true, data will be saved after the analysis
    x_label : str
            X label of the data when plotted
    y_label : str
            Y label of the data when plotted
    analysis_name : str
            Name of the analysis
    loaded_property : str
            Property loaded from the database for the analysis
    batch_loop : int
            Number of ensembles in each batch
    time : np.array
            Array of the time.
    correlation_time : int
            Correlation time of the property being studied. This is used to ensure ensemble sampling is only performed
            on uncorrelated samples. If this is true, the error extracted form the calculation will be correct.
    """

    def __init__(self, obj, plot=False, data_range=500, x_label='Time (s)', y_label=r'JACF ($C^{2}\cdot m^{2}/s^{2}$)',
                 save=True, analysis_name='green_kubo_ionic_conductivity', correlation_time=1):
        """

        Attributes
        ----------
        obj :  object
                Experiment class to call from
        plot : bool
                if true, plot the data
        data_range :
                Number of configurations to use in each ensemble
        save :
                If true, data will be saved after the analysis
        x_label : str
                X label of the data when plotted
        y_label : str
                Y label of the data when plotted
        analysis_name : str
                Name of the analysis
        """

        # update parent class
        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name, parallel=True,
                         correlation_time=correlation_time)

        self.loaded_property = 'Ionic_Current'  # property to be loaded for the analysis
        self.tensor_choice = False  # Load data as a tensor
        self.database_group = 'ionic_conductivity'  # Which database group to save the data in

        # Check for unwrapped coordinates and unwrap if not stored already.
        with hf.File(os.path.join(obj.database_path, 'database.hdf5'), "r+") as database:
            # Unwrap the positions if they need to be unwrapped
            if self.loaded_property not in database:
                self.parent.perform_transformation('IonicCurrent', calculator=self)

    def _autocorrelation_time(self):
        """
        calculate the current autocorrelation time for correct sampling
        """
        pass

    def _calculate_ionic_conductivity(self):
        """
        Calculate the ionic conductivity in the system
        """

        # Calculate the prefactor
        numerator = (elementary_charge ** 2) * (self.parent.units['length'] ** 2)
        denominator = 3 * boltzmann_constant * self.parent.temperature * self.parent.volume * \
                      (self.parent.units['length'] ** 3) * self.data_range * self.parent.units['time']
        prefactor = numerator / denominator

        db_path = join_path(self.loaded_property, self.loaded_property)

        sigma = []
        parsed_autocorrelation = tf.zeros(self.data_range, dtype=tf.float64)

        for i in range(int(self.n_batches['Parallel'])):  # loop over batches
            batch = self.load_batch(i, path=db_path)

            def generator():
                for start_index in range(int(self.batch_loop)):
                    start = start_index + self.correlation_time
                    stop = start + self.data_range
                    yield batch[start:stop]

            dataset = tf.data.Dataset.from_generator(
                generator,
                output_signature=tf.TensorSpec(shape=(self.data_range, 3), dtype=tf.float64)
            )
            dataset = dataset.map(self.convolution_op, num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                  deterministic=False)

            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

            # for frame in tqdm(dataset.batch(self.data_range), total=int(self.batch_loop / self.data_range)):
            for frame in tqdm(dataset, total=int(self.batch_loop), smoothing=0.1):
                parsed_autocorrelation += frame[self.data_range - 1:]
                sigma.append(np.trapz(frame[self.data_range - 1:], x=self.time))

        sigma = prefactor * tf.constant(sigma)
        parsed_autocorrelation = parsed_autocorrelation / max(parsed_autocorrelation)

        # update the experiment class
        self._update_properties_file(data=[str(np.mean(sigma)), str((np.std(sigma) / np.sqrt(len(sigma))))])

        plt.plot(self.time * self.parent.units['time'], parsed_autocorrelation)  # Add a plot

        parsed_autocorrelation /= max(parsed_autocorrelation)  # Get the normalized autocorrelation plot data

        # save the data if necessary
        if self.save:
            self._save_data(f'{self.analysis_name}', [self.time, parsed_autocorrelation])

        if self.plot:
            self._plot_data()  # Plot the data if necessary

    def run_analysis(self):
        """
        call relevant methods and run analysis
        """

        self._autocorrelation_time()  # get the autocorrelation time
        self.collect_machine_properties()  # collect machine properties and determine batch size
        self._calculate_batch_loop()  # Update the batch loop attribute
        status = self._check_input()  # Check for bad input
        if status == -1:
            return
        else:
            self._calculate_ionic_conductivity()  # calculate the ionic conductivity

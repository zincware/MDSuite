"""
Class for the calculation of the einstein diffusion coefficients.

Summary
-------
Description: This module contains the code for the Einstein diffusion coefficient class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.einstein_diffusion_coefficients method.
The methods in class can then be called by the Experiment.einstein_diffusion_coefficients method and all necessary
calculations performed.
"""

# Python standard packages
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import warnings
import tensorflow as tf
import h5py as hf

# Import user packages
from tqdm import tqdm

# Import MDSuite packages
import mdsuite.utils.meta_functions as meta_functions
from mdsuite.calculators.calculator import Calculator

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class EinsteinDiffusionCoefficients(Calculator):
    """
    Class for the Einstein diffusion coefficient implementation

    Description: This module contains the code for the Einstein diffusion coefficient class. This class is called by the
    Experiment class and instantiated when the user calls the Experiment.einstein_diffusion_coefficients method.
    The methods in class can then be called by the Experiment.einstein_diffusion_coefficients method and all necessary
    calculations performed.

    Attributes
    ----------
    obj :  object
            Experiment class to call from
    plot : bool
            if true, plot the data
    singular : bool
            If true, calculate the singular diffusion coefficients
    distinct : bool
            If true, calculate the distinct diffusion coefficient
    species : list
            Which species to perform the analysis on
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

    def __init__(self, obj, plot=True, singular=True, distinct=False, species=None, data_range=200, save=True,
                 x_label='Time (s)', y_label='MSD (m^2/s)', analysis_name='einstein_diffusion_coefficients'):
        """

        Parameters
        ----------
        obj :  object
                Experiment class to call from
        plot : bool
                if true, plot the data
        singular : bool
                If true, calculate the singular diffusion coefficients
        distinct : bool
                If true, calculate the distinct diffusion coefficient
        species : list
                Which species to perform the analysis on
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

        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name)  # parse to the parent class

        self.loaded_property = 'Unwrapped_Positions'  # Property to be loaded
        self.batch_loop = None                        # Number of ensembles in each batch
        self.parallel = False                         # Set the parallel attribute
        self.tensor_choice = True                     # Load data as a tensor

        self.singular = singular                      # calculate singular diffusion if true
        self.distinct = distinct                      # calculate distinct diffusion if true
        self.species = species                        # Which species to calculate the diffusion for

        # Time array
        self.time = np.linspace(0.0, self.data_range * self.parent.time_step * self.parent.sample_rate, self.data_range)
        self.correlation_time = 100  # correlation time TODO: do not hard code this.

        # Check for unwrapped coordinates and unwrap if not stored already.
        with hf.File(f"{obj.storage_path}/{obj.analysis_name}/{obj.analysis_name}.hdf5", "r+") as database:
            for item in species:
                # Unwrap the positions if they need to be unwrapped
                if "Unwrapped_Positions" not in database[item]:
                    print("Unwrapping coordinates")
                    obj.unwrap_coordinates(species=[item])  # perform the coordinate unwrapping.
                    print("Coordinate unwrapping finished, proceeding with analysis")

    def _autocorrelation_time(self):
        """
        Calculate positions autocorrelation time

        When performing this analysis, the sampling should occur over the autocorrelation time of the positions in the
        system. This method will calculate what this time is and sample over it to ensure uncorrelated samples.
        """
        pass

    def _optimize_data_range(self):
        """
        Optimize the data range of a system using the Einstein method of calculation.

        Returns
        -------
        Updates the data_range attribute of the class state
        """
        pass

    def _single_diffusion_coefficients(self):
        """
        Calculate singular diffusion coefficients

        Implement the Einstein method for the calculation of the singular diffusion coefficient. This is performed
        using unwrapped coordinates, generated by the unwrap method. From these values, the mean square displacement
        of each atom is calculated and averaged over all the atoms in the system.
        """

        # Loop over each atomic species to calculate self-diffusion
        for item in list(self.species):
            msd_array = np.zeros(self.data_range)  # define empty msd array

            # Calculate the prefactor
            numerator = self.parent.units['length'] ** 2
            denominator = (self.parent.units['time'] * len(self.parent.species[item]['indices'])) * 6
            prefactor = numerator / denominator

            # Construct the MSD function
            for i in tqdm(range(int(self.n_batches['Serial'])), ncols=70):
                batch = self._load_batch(i, [item])  # load a batch of data
                for start_index in range(self.batch_loop):
                    start = start_index*self.data_range + self.correlation_time
                    stop = start + self.data_range
                    window_tensor = batch[:, start:stop]

                    # Calculate the msd
                    msd = (window_tensor - (
                        tf.repeat(tf.expand_dims(window_tensor[:, 0], 1), self.data_range, axis=1))) ** 2

                    # Sum over trajectory and then coordinates and apply averaging and prefactors
                    msd = prefactor * tf.reduce_sum(tf.reduce_sum(msd, axis=0), axis=1)
                    msd_array += np.array(msd)  # Update the averaged function

            msd_array /= int(self.n_batches['Serial'])*self.batch_loop  # Apply the batch/loop average

            popt, pcov = curve_fit(meta_functions.linear_fitting_function, self.time, msd_array)
            self.parent.diffusion_coefficients["Einstein"]["Singular"][item] = [popt[0], np.square(np.diag(pcov))[0]]

            # Update the plot if required
            if self.plot:
                plt.plot(np.array(self.time) * self.parent.units['time'], msd_array, label=item)

            # Save the array if required
            if self.save:
                self._save_data(f"{item}_{self.analysis_name}", [self.time, msd_array])

        # Save a figure if required
        if self.plot:
            self._plot_data()

    def _distinct_diffusion_coefficients(self):
        """
        Calculate the Distinct Diffusion Coefficients

        Use the Einstein method to calculate the distinct diffusion coefficients of the system from the mean
        square displacement, as calculated from different atoms. This value is averaged over all the possible
        combinations of atoms for the best fit.
        """
        # TODO: Implement this function
        raise NotImplementedError

    def run_analysis(self):
        """
        Run a diffusion coefficient analysis

        In order to full calculate diffusion coefficients from a simulation, one should perform a two-stage error
        analysis on the data. The two steps are performed at the same time. The first error contribution comes from
        the fit error of the scipy fit package. The second is calculated by averaging the diffusion coefficient
        calculated at start times, taken over correlation times.
        """

        self._autocorrelation_time()                    # get the autocorrelation time
        self._collect_machine_properties()              # collect machine properties and determine batch size
        self._calculate_batch_loop()                    # Update the batch loop attribute
        if self.singular:
            self._single_diffusion_coefficients()       # calculate the singular diffusion coefficients
        if self.distinct:
            self._distinct_diffusion_coefficients()     # calculate the distinct diffusion coefficients

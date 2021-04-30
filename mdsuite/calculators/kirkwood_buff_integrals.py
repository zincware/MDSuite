"""
Class for the calculation of the coordinated numbers
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Union
from mdsuite.database.properties_database import PropertiesDatabase
from mdsuite.database.database_scheme import SystemProperty

# MDSuite imports
from mdsuite.utils.exceptions import *
from mdsuite.calculators.calculator import Calculator


class KirkwoodBuffIntegral(Calculator):
    """
    Class for the calculation of the Kikrwood-Buff integrals

    Attributes
    ----------
    experiment : class object
                        Class object of the experiment.
    plot : bool (default=True)
                        Decision to plot the analysis.
    save : bool (default=True)
                        Decision to save the generated tensor_values arrays.

    data_range : int (default=500)
                        Range over which the property should be evaluated. This is not applicable to the current
                        analysis as the full rdf will be calculated.
    x_label : str
                        How to label the x axis of the saved plot.
    y_label : str
                        How to label the y axis of the saved plot.
    analysis_name : str
                        Name of the analysis. used in saving of the tensor_values and figure.
    file_to_study : str
                        The tensor_values file corresponding to the rdf being studied.
    data_files : list
                        list of files to be analyzed.
    rdf = None : list
                        rdf tensor_values being studied.
    radii = None : list
                        radii tensor_values corresponding to the rdf.
    species_tuple : list
                        A list of species combinations being studied.
    pomf : list
                        List of tensor_values of the potential of mean-force for the current analysis.
    """

    def __init__(self, experiment, plot=True, save=True, data_range=1, export: bool = False):
        """
        Python constructor for the class

        Parameters
        ----------
        experiment : class object
                        Class object of the experiment.
        plot : bool (default=True)
                            Decision to plot the analysis.
        save : bool (default=True)
                            Decision to save the generated tensor_values arrays.

        data_range : int (default=500)
                            Range over which the property should be evaluated. This is not applicable to the current
                            analysis as the full rdf will be calculated.
        x_label : str
                            How to label the x axis of the saved plot.
        y_label : str
                            How to label the y axis of the saved plot.
        analysis_name : str
                            Name of the analysis. used in saving of the tensor_values and figure.
        """

        super().__init__(experiment, plot, save, data_range, export=export)
        self.file_to_study = None
        self.data_files = []
        self.rdf = None
        self.radii = None
        self.species_tuple = None
        self.kb_integral = None
        self.database_group = 'Kirkwood_Buff_Integral'
        self.x_label = r'r ($\AA$)'
        self.y_label = r'$G(\mathbf{r})$'
        self.analysis_name = 'Kirkwood-Buff_Integral'

        self.post_generation = True

    def _autocorrelation_time(self):
        """
        Not needed in this analysis
        """
        raise NotApplicableToAnalysis

    def _get_rdf_data(self) -> list:
        """
        Fill the data_files list with filenames of the rdf tensor_values
        """
        database = PropertiesDatabase(name=os.path.join(self.experiment.database_path, 'property_database'))

        return database.load_data({"property": "RDF"})

    def _load_rdf_from_file(self, system_property: SystemProperty):
        """
        Load the raw rdf tensor_values from a directory

        Parameters
        ----------
        system_property: SystemProperty
        """

        radii = []
        rdf = []
        for _bin in system_property.data:
            radii.append(_bin.x)
            rdf.append(_bin.y)

        self.radii = np.array(radii)
        self.rdf = np.array(rdf)

    def _calculate_kb_integral(self):
        """
        calculate the Kirkwood-Buff integral
        """

        self.kb_integral = []  # empty the integration tensor_values

        for i in range(1, len(self.radii)):
            self.kb_integral.append(4*np.pi*(np.trapz((self.rdf[1:i] - 1)*(self.radii[1:i])**2, x=self.radii[1:i])))

    def run_post_generation_analysis(self):
        """
        Calculate the potential of mean-force and perform error analysis
        """

        for data in self._get_rdf_data():  # Loop over all existing RDFs
            self.species_tuple = "_".join([subject.subject for subject in data.subjects])
            self.data_range = data.data_range

            self._load_rdf_from_file(data)  # load the tensor_values from it

            self._calculate_kb_integral()    # Integrate the rdf and calculate the KB integral

            # Plot if required
            if self.plot:
                plt.plot(self.radii[1:], self.kb_integral, label=f"{self.species_tuple}")
                self._plot_data(title=f"{self.analysis_name}_{self.species_tuple}")

            # TODO what to save!
            # if self.save:
            #     self._save_data(name=self._build_table_name(self.species_tuple),
            #                     data=self._build_pandas_dataframe(self.radii[1:], self.kb_integral))
            # if self.export:
            #     self._export_data(name=self._build_table_name(self.species_tuple),
            #                       data=self._build_pandas_dataframe(self.radii[1:], self.kb_integral))

    def _calculate_prefactor(self, species: Union[str, tuple] = None):
        """
        calculate the calculator pre-factor.

        Parameters
        ----------
        species : str
                Species property if required.
        Returns
        -------

        """
        raise NotImplementedError

    def _apply_operation(self, data, index):
        """
        Perform operation on an ensemble.

        Parameters
        ----------
        One tensor_values range of tensor_values to operate on.

        Returns
        -------

        """
        raise NotImplementedError

    def _apply_averaging_factor(self):
        """
        Apply an averaging factor to the tensor_values.
        Returns
        -------
        averaged copy of the tensor_values
        """
        raise NotImplementedError

    def _post_operation_processes(self, species: Union[str, tuple] = None):
        """
        call the post-op processes
        Returns
        -------

        """
        raise NotImplementedError

    def _update_output_signatures(self):
        """
        After having run _prepare managers, update the output signatures.

        Returns
        -------

        """
        raise NotImplementedError
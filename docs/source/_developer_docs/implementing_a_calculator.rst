Implementing a Calculator
-------------------------
Whilst MDSuite offers a large selection of calculators, we understand that it is
quite likely we have missed at least one.
For this reason, we wrote the calculator class and the actual calculators in a modular,
transferable way such that you can easily implement your own while getting the most out
of the memory safety and data-set capabilities that make MDSuite a high performance
engine.

In this documentation series we will walk through the main stages of a calculator
and highlight differences you may face using real examples.

Imports
=======
The first step to many python programs are the import statements.
In the case of an MDSuite calculator the important ones are as follows:

.. code-block:: python

   import logging
   from mdsuite.calculators import Calculator, call
   from dataclasses import dataclass
   from mdsuite.database import simulation_properties

   log = logging.getLogger(__name__)

Let's go through them one by one.

* logging is the library you will want to use to log information. Typically, when you
  want to output some information you might use a print statement. However, by using
  for example log.INFO or log.WARNING you can better utilize Python software frameworks.
* Calculator is the parent class for all of our calculators. You need to import this
  and inherit from it so that you can access the simulation data in a batched way.
* call is a decorator method we have implemented which helps to take care of some
  commonly called methods between calculators such as saving data into the SQL database.
* dataclass will be used to define a dataclass for the properties that are saved into
  the MDSuite project database.
* simulation_properties is a dataclass of all the different properties that can be
  stored in an MDSuite simulation database. If your calculator is using data that you
  have specially made and added, you will need to include the name of the HDF5 database
  group to this dataclass. This will all be covered in the
  'Implementing a Transformation' and 'Implementing a File Reader' tutorials.

Dataclass
=========
One of the great things about MDSuite is that it stores the outcome of all computations
in an SQL database along with the meta-data of the calculation.
In order for this to happen, the calculator needs to know what should be saved.
We do this using a dataclass for each calculator, the contents of which will end up
in the project database.

.. code-block:: python

   @dataclass
   class Args:
       data_range: int
       correlation_time: int
       atom_selection: np.s_
       tau_values: np.s_
       species: list

Here we have shown the dataclass from our Einstein diffusion coefficients.
In this calculator, we want to store the data_range over which the function was measured
, the correlation time used, the atoms selected for the computation, the time steps used
, and the species studied in the analysis.
Consider now the dataclass for the radial distribution function:

.. code-block:: python

   @dataclass
   class Args:
       number_of_bins: int
       number_of_configurations: int
       atom_selection: np.s_
       cutoff: float
       start: int
       stop: int
       species: list

In this case, any of the information related with data ranges or correlation times is
removed in favour of knowing how many configurations and bins were used in the analysis.
This isn't the end of the story for the SQL database, there is one more part that we
will discuss shortly.

Declaration and __init__
========================
We can now define the class.
In reality, the __init__ method of MDSuite calculators is not overly useful as we rely
mostly on the __call__ method.
However, as best practice dictates, we must still declare our class variables in the
__init__ and for your own benefit, should document them there.

.. code-block:: python

   class NewCalculator(Calculator):
       """
       Class for the Einstein diffusion coefficient implementation

       Notes
       -----

       Attributes
       ----------

       See Also
       --------

       Examples
       --------
       """

       def __init__(self, **kwargs):
           """

           Parameters
           ----------

           """

           super().__init__(**kwargs)
           self.scale_function = {"linear": {"scale_factor": 150}}
           self.loaded_property = simulation_properties.unwrapped_positions
           self.database_group = "Diffusion_Coefficients"
           self.x_label = r"$ x label $"
           self.y_label = r"$ y label $"
           self.result_keys = ["diffusion_coefficient", "uncertainty"]
           self.result_series_keys = ["time", "msd"]
           self.analysis_name = "Name of Analysis"
           self.optimize = None
           self.msd_array = None
           self.tau_values = None
           self._dtype = tf.float64

           log.info("starting ... Computation")

This barebones example was taken from the einstein diffusion coefficient calculator
where we have stripped out a lot of doc-strings and removed some string in place of a
more descriptive option.

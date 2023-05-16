Quickstart Guide
================
In this section we discuss briefly the outline of the MDSuite project and how to get
started.

MDSuite architecture
--------------------
MDSuite was designed to feel like a natural experimental design for computational
scientists.
The figure below outlines the different elements of the software and how they interact
with one another.

 .. figure:: images/project_structure.png
      :alt: MDSuite project structure
      :class: with-shadow
      :width: 400px

      Outline of the MDSuite project structure.

The largest component of the MDSuite library is the `Project`.
This class is named so as to mimic any computational project such as the study of molten
salt properties with respect to temperature or a comparison of protein models.
A project is constructed in MDSuite as follows:

.. code-block:: python

   import mdsuite as mds

   my_project = mds.Project(name="My_Cool_Project")


With this call, a new project has been created.
Of course, there is no data here yet.
Data enters in the form of `Experiments`.
An experiment is a specific simulation that has been performed as a part of the project.
For example, in the case of a molten salt study over temperatures, there would be one
experiment for a 1400K simulation and another experiment for a 1500K simulation.
Experiments can be added to a project directly as follows:

.. code-block:: python

   experiment = my_project.add_experiment(
        name="NaCl_example_data",
        timestep=0.002,
        temperature=1400.0,
        units="metal",
        simulation_data="NaCl_gk_i_q.lammpstraj",
    )

In this case, data has been directly added to the experiment from a lammps simulation.
MDSuite can read in file formats from most common simulations engines and is always
working to extend its applicability.
With this, you now have a project as well as one experiment added to it.
Of course, you now what to study the systems you have added.
This can be called directly from the `project` by:

.. code-block:: python

   calc_data = my_project.run.CalculatorName(**calc_params)


When this code is run, the calculation will be performed over all available experiments
and the results will be stored in the SQL database created when the project was constructed.
The interesting thing happens next.
If you know re-run the exact same calculator with the same parameters, MDSuite will
recognize that this calculation has already been performed and will return the data to
you rather than re-running it.
Of course, if the parameters have changed, the calculation will be re-run.

Looking back at the addition of the experiment, you will see that the addition of an
experiment returned an object.
This allows you to run a calculator directly on one experiment rather than on all attached
to the `project`.
There are many more features present in the code that we will discuss in other parts of
the documentation.

Quickstart Guide
================
Here we outline briefly how to get started with MDSuite, namely importing the package and creating an Experiment.

Installation
------------

With pip:

.. code-block:: console

    $ cd MDSuite
    $ pip3 install . --user

Usage
-----

As MDSuite is installed simply as a python package, once installed, it can be called from any 
python program you are writing. This is done in the usual way, 

.. code-block:: python
        
        import mdsuite
        import mdsuite as mds
        from mdsuite import ...

which will load whichever modules are desired. Once you have import the module, you will need 
to begin your first analysis. This can be done as follows.

.. code-block:: python
        
        import mdsuite as mds

        test_project =mds.Experiment(analysis_name='name of the experiment',
                                     time_step='time_step',
                                     temperature='temperature of the simulation',
                                     units='LAMMPS units keyword e.g. real',
                                     cluster_mode='Are you on a cluster?',
                                     storage_path='where to save data')

Once this has been constructed, and the database has been built, you will have access to all of 
the methods currently available in the mdsuite package. For information about what these are, go 
and check out the examples directory and run some for yourself.

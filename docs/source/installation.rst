Installation
------------

Installation of MDSuite is currently only possible from the git repository.
Once all of the documentation is up, we will move this over to PyPi and conda.

From Source
===========

From the directory in which you wish to store MDSuite, run

.. code-block:: bash
        
        git clone https://github.com/SamTov/MDSuite.git

Once the directory has finished cloning, change into the MDSuite directory,
and run

.. code-block:: bash
        
   pip3 install . --user

Or if you want to install as developer:

.. code-block:: bash

   pip3 install -e . --user

which will install the mdsuite package as a python library. Once this has
been done, you can simply call mdsuite from any of your python projects with

.. code-block:: python
        
   import mdsuite

or with an alias you prefer. We usually run with

.. code-block:: python

   import mdsuite as mds

Once you have installed the package, go and check out the tutorials directory
to see how you can start using it to analyze your own simulations.

.. toctree::
    :maxdepth: 1

    _installation/mac_silicon

|madewithpython|

Introduction
------------

MDSuite is a software designed specifically for the molecular dynamics community to enable fast, reliable, and simple 
calculations from simulation data. 

If you want to start learning about the code, you can check out the docs `here <https://mdsuite.readthedocs.io/en/latest/>`.

**NOTE:** This README is designed for the release. Anything PyPi related is not yet functioning as the code is in
development.

Installation
============

There are several way to install MDSuite depending on what you would like from it. One can simply installing using 
PyPi as **Not Currently available**

.. code-block:: bash

   $ pip install mdsuite

If you would like to install it from source then you can clone the repository by running

.. code-block:: bash

   $ git clone https://github.com/SamTov/MDSuite.git

Once you have cloned the repository, depending on whether you prefer conda or straight pip, you should follow the 
instructions below.

Installation with pip
*********************

.. code-block:: bash

    $ cd MDSuite
    $ pip install .


Installation with conda
***********************

.. code-block:: bash

    $ cd MDSuite
    $ conda create -n MDSuite python=3.8
    $ conda activate MDSuite
    $ pip install .

Documentation
=============

There is a live version of the documentation hosted `here <https://mdsuite.readthedocs.io/en/latest/>`.
If you would prefer to have a local copy, it can be built using sphinx by following the instructions below.

.. code-block:: bash

   $ cd MDSuite/docs
   $ make html
   $ cd build/html
   $ firefox index.html

HINT
====

Check out the MDSuite code through a jupyter notebook for a more user friendly experience. You can take full advantage
of the autocomplete features that are available for the calculators.

![Alt Text](docs/source/images/test_einstein_record_3.gif)

.. badges

.. |madewithpython| image:: https://img.shields.io/badge/Made%20With-Python-blue.svg
    :alt: Made with Python

.. |build| image:: https://img.shields.io/badge/Build-Passing-green.svg
    :alt: Build tests passing
    :target: https://github.com/zincware/MDSuite/blob/readme_badges/.github/workflows/python-package.yaml

.. |docs| image:: https://img.shields.io/badge/Build-Passing-green.svg
    :alt: Build tests passing
    :target: https://github.com/zincware/MDSuite/blob/readme_badges/.github/workflows/doc.yml

.. |license| image:: https://img.shields.io/badge/License-EPL-green.svg
    :alt: Project license
    :target: https://www.gnu.org/licenses/quick-guide-gplv3.en.html
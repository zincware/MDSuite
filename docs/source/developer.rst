Developer documentation
=======================
Welcome to the developer documentation. We are glad you are interested in helping out with the code. Here you will find
some resources to help you get started and avoid making some of the same mistakes we did when we started out.

.. toctree::
    :maxdepth: 1

    _developer_docs/faq
    _developer_docs/implementing_calculators

Code Style
-----------

MDSuite uses `Black <https://github.com/psf/black>`_, `Isort <https://github.com/PyCQA/isort>`_ and `ruff <https://github.com/charliermarsh/ruff>`_.
We provide a pre-commit hook to check these requirements.
One can install the hook via

.. code-block::

    pip install pre-commit
    pre-commit install

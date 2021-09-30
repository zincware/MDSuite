"""
MDSuite: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
"""
import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MDSuite",
    version="0.0.1",
    author="Samuel Tovey",
    author_email="tovey.samuel@gmail.com",
    description="A postprocessing tool for molecular dynamics simulations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SamTov/MDSuite",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)",
        "Operating System :: OS Independent",
    ],
    package_data={"": ["form_fac_coeffs.csv"]},
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "tensorflow>=2.5",
        "h5py",
        "pysmiles",
        "matplotlib",
        "scipy",
        "tqdm",
        "psutil>=5.6.6",
        "numpy",
        "gputil",
        "pubchempy",
        "PyYAML>=5.4",
        "scooby",
        "sqlalchemy >= 1.4",
        "pandas >= 1.0.0",
        "tensorflow_probability",
        "open3d",
        "bokeh>=2.4.0",
    ],
)

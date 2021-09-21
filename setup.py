"""
Setup.py module for the MDSuite package.
"""
import setuptools
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]


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
    package_data={'': ['form_fac_coeffs.csv']},
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=requirements,
)

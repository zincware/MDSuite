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
    package_data={'': ['form_fac_coeffs.csv']},
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=[
        'tensorflow',
        'h5py',
        'pysmiles',
        'matplotlib',
        'scipy',
        'tqdm',
        'psutil>=5.6.6',
        'numpy',
        'gputil',
        'pubchempy',
        'PyYAML>=5.4',
        'scooby',
        'sqlalchemy >= 1.4',
        'pandas >= 1.0.0',
        'tensorflow_probability'
    ]
)

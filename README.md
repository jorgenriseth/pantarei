# Panta Rei
A collection of utilities for fluid flow modelling using FeNICS.

## Dependencies
The package depends on [FEniCS](https://fenicsproject.org/download/) and [SVMTK](https://github.com/SVMTK/SVMTK), which is not installed by default, since they are not available on PyPI. They are most easily installed using conda, by running the following commands: 
```bash
conda create -n fenicsproject -c conda-forge/label/fenics-dev  -c conda-forge fenics SVMTK h5py
conda activate fenicsproject
```
`h5py` should also be installed, or else there might be conflicts between the version of `h5py` that is installed with `fenics` and the one that is installed by `meshio` (listed as a pip-installable dependency).

## Installation
The pantarei package itself may be installed using `pip`,
```bash
pip install pantarei
```
or by cloning the repository and running the following command in the root directory:
```bash
pip install -e .
```

## Running tests
Tests depends on the Pixi, a python package manager tightly integrated with conda. It can be installed by running
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```
The tests can be run by running the following command in the root directory:
```bash
pixi run test
```
For more information on Pixi see [https://pixi.sh/](https://pixi.sh/), or [this guide by Eric Ma](https://ericmjl.github.io/blog/2024/8/16/its-time-to-try-out-pixi).

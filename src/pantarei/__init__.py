import importlib.util
from importlib.metadata import version

__version__ = version(__package__)

for package in ["dolfin", "SVMTK"]:
    try:
        importlib.import_module(package)
    except ImportError:
        raise ImportError(f"""
            Module {package} is not installed.
            Please install it before using pantarei.
            See README.md for more information.
        """)


import pantarei.computers
import pantarei.fenicsstorage
import pantarei.meshprocessing
import pantarei.projectors
import pantarei.solvers
import pantarei.timekeeper
import pantarei.utils
from pantarei.boundary import *
from pantarei.computers import *
from pantarei.fenicsstorage import *
from pantarei.interpolator import *
from pantarei.io_utils import *
from pantarei.meshprocessing import *
from pantarei.mms import *
from pantarei.solvers import *
from pantarei.timekeeper import *
from pantarei.utils import *

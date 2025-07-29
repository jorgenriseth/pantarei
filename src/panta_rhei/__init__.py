import importlib.util
from importlib.metadata import version

__version__ = version("panta-rhei")

for package in ["dolfin", "SVMTK"]:
    try:
        importlib.import_module(package)
    except ImportError:
        raise ImportError(f"""
            Module {package} is not installed.
            Please install it before using panta_rhei.
            See README.md for more information.
        """)
import ufl_legacy as ufl

from panta_rhei.boundary import *
from panta_rhei.computers import *
from panta_rhei.fenicsstorage import *
from panta_rhei.interpolator import *
from panta_rhei.io_utils import *
from panta_rhei.meshprocessing import *
from panta_rhei.mms import *
from panta_rhei.solvers import *
from panta_rhei.timekeeper import *
from panta_rhei.utils import *

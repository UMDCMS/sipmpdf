from . import version

__version__ = version.__version__

import sys

if sys.version_info.major < 3:
  import warnings
  warnings.warn("Only supports python3!")

from . import kernel
from . import functions

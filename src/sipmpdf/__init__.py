from . import version

__version__ = version.__version__

import os
import sys

if sys.version_info.major < 3:
    import warnings

    warnings.warn("Only supports python3!")

# Setting tensorflow fixing the limit to 4GB memory limit
import tensorflow

gpus = tensorflow.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            memory_limit = os.getenv("_SIPMPDF_GPU_MEMORY_LIMIT")
            try:
                memory_limit = int(memory_limit)
            except:
                memory_limit = 1024 * 4

            tensorflow.config.experimental.set_virtual_device_configuration(
                gpu,
                [
                    tensorflow.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit
                    )
                ],
            )
    except RuntimeError as e:
        print(e)

from . import functions, generate,  kernel, pdf

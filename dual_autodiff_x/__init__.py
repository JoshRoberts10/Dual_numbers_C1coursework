import os
import sys

try:
    from .dual import Dual
except ImportError as e:
    current_dir = os.path.dirname(__file__)
    sys.stderr.write(
        f"Failed to import 'dual'. Directory contents: {os.listdir(current_dir)}\n"
        f"Current directory: {current_dir}\n"
        f"Original error: {e}\n"
    )
    raise ImportError(
        f"Could not import 'dual' from 'dual_autodiff_x'. Ensure the package "
        f"and its compiled extensions are installed correctly."
    )

__all__ = ["Dual"]



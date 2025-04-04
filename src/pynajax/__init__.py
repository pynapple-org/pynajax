from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _get_version

try:
    __version__ = _get_version("pynajax")
except _PackageNotFoundError:
    # package is not installed
    pass
# from .jax_core_bin_average import bin_average
# from .jax_core_convolve import convolve

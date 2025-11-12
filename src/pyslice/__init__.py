# here we read out the version info set in pyproject.toml
from importlib.metadata import version
__version__ = version("pyslice")
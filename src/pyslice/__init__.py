# here we read out the version info set in pyproject.toml
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__))) # see https://stackoverflow.com/questions/16981921/relative-imports-in-python-3, this allows sys.path.insert calls in the tests to work when pyslice has not been pip installed
from importlib.metadata import version
__version__ = version("pyslice")
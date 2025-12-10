# here we read out the version info set in pyproject.toml
try:
    from importlib.metadata import version
    __version__ = version("pyslice")
except Exception:
    __version__ = "dev"

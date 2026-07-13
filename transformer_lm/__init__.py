import importlib.metadata

try:
    __version__ = importlib.metadata.version("transformer_lm")
except importlib.metadata.PackageNotFoundError:
    pass

from contextlib import _GeneratorContextManager

__version__: str

def config_context(*, transform_output: str | None = None) -> _GeneratorContextManager[None]: ...

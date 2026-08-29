"""Package exceptions."""

class GP3MLError(ValueError):
    """Raised when a gp3ml scientific or validation contract is violated."""

class OptionalDependencyError(ImportError):
    """Raised when an explicitly requested optional engine is unavailable."""

"""Enterprise stub utilities for agents that require external configuration."""


class NotConfiguredError(RuntimeError):
    """Raised when an agent requires configuration that is not set."""

    pass

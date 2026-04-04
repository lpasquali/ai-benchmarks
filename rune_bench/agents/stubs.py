"""Enterprise stub utilities for agents that require external configuration.

Agents that are registered in the built-in map but depend on commercial
licences, SaaS API keys, or specialised infrastructure should raise
:class:`NotConfiguredError` at construction time when the required
configuration is absent.  This gives callers a clear, actionable message
instead of an opaque import or runtime failure deep in vendor code.
"""


class NotConfiguredError(RuntimeError):
    """Raised when an agent requires configuration that is not set."""

    pass

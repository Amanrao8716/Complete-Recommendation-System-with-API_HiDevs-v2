"""Exception hierarchy shared by every layer.

Catch :class:`RecsysError` to handle any error raised *deliberately* by
this project; unexpected bugs (``KeyError``, ``AttributeError``...) are
left alone so they surface as 500s with a traceback in the logs.

The classes also inherit from the matching built-in (``ValueError``,
``LookupError``) so older ``except ValueError`` code keeps working.
"""

from __future__ import annotations


class RecsysError(Exception):
    """Base class for all intentional errors in this project."""


class ValidationError(RecsysError, ValueError):
    """An argument or payload has the wrong type, shape or range."""


class NotFoundError(RecsysError, LookupError):
    """A requested entity (user, content...) does not exist."""


class ConfigError(RecsysError, ValueError):
    """Configuration is missing, malformed or out of range."""

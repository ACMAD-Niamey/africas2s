"""Model Context Protocol server for africas2s (``pip install 'africas2s[mcp]'``).

Run it with ``africas2s-mcp`` (stdio) or ``python -m africas2s.mcp``. Tools
mirror the public verbs (``downscale``, ``optimize``, ``calibrate``,
``ensemble``, ``skill``, tercile conversion, and the two core map plots) and
exchange data as NetCDF files on disk, so outputs from a data layer such as
the acmaddl MCP server can be handed over by path.
"""

from .server import mcp, main  # noqa: F401

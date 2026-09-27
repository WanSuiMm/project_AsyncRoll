"""Optional NVTX process ranges for correlating AsyncRoll events in Nsight.

Range labels are stable, low-cardinality names such as ``MODEL_REQUEST``.
Numeric identities are stored in the NVTX payload, separate from the message,
so request IDs do not create a distinct registered message for every range.
NVTX's built-in payload support covers integers and floats without NumPy.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


class Profiler:
    """Start and end explicit NVTX ranges, or do nothing when disabled.

    ``identity`` should be an integer or float (typically a request or CPU job
    ID). It is recorded as an NVTX payload while ``label`` remains the range
    message. Explicit start/end handles allow overlapping asyncio tasks and
    ending a range from a different thread; push/pop ranges are thread-local
    and are therefore unsuitable here.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._nvtx: Any = None
        self._open: dict[object, Any] = {}
        if self.enabled:
            try:
                self._nvtx = import_module("nvtx")
            except ImportError as exc:
                raise ImportError(
                    "NVTX profiling was requested, but the optional 'nvtx' "
                    "package is unavailable. Install AsyncRoll's profiling "
                    "extra (or install the 'nvtx' package) to enable it."
                ) from exc

    def start(self, label: str, identity: int | float | None = None) -> object | None:
        """Start a range and return its opaque handle; disabled mode is a no-op."""
        if not self.enabled:
            return None
        if not isinstance(label, str) or not label:
            raise ValueError("NVTX range label must be a non-empty string")
        if identity is not None and (
            isinstance(identity, bool) or not isinstance(identity, (int, float))
        ):
            raise TypeError("NVTX range identity must be an integer or float")

        options = {} if identity is None else {"payload": identity}
        range_id = self._nvtx.start_range(message=label, **options)
        handle = object()
        self._open[handle] = range_id
        return handle

    def end(self, handle: object | None) -> None:
        """End a range once; unknown, already-ended, and disabled handles no-op."""
        if not self.enabled or handle is None:
            return
        range_id = self._open.pop(handle, None)
        if range_id is not None:
            self._nvtx.end_range(range_id)

    def close(self) -> None:
        """End all ranges left open, for orderly shutdown and error cleanup."""
        first_error: BaseException | None = None
        for handle in tuple(self._open):
            try:
                self.end(handle)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

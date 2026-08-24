"""Small asyncio boundaries shared by host-independent workflows."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any


async def run_sync_transaction[ResultT](
    function: Callable[..., ResultT],
    /,
    *args: Any,
    **kwargs: Any,
) -> ResultT:
    """Do not return cancellation while an already-started sync transaction is unresolved."""
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        try:
            await worker
        except Exception:  # noqa: BLE001
            pass
        raise

import asyncio
from typing import Generator, AsyncGenerator

class Sentinel:
    pass

async def _sync_to_async_generator(
    gen: Generator[str, None, None]
) -> AsyncGenerator[str, None]:
    """
    Converts a sync generator into an async generator
    without blocking the event loop.
    """
    loop = asyncio.get_running_loop()
    sentinel = Sentinel()

    def safer_next():
        try:
            return next(gen)
        except StopIteration:
            return sentinel

    while True:
        token = await loop.run_in_executor(None, safer_next)
        if isinstance(token, Sentinel):
            break
        yield token

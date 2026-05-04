import asyncio
import time


class GeminiRateLimiter:
    """Gemini 2.5 Flash free tier: 10 RPM → 12s safe gap."""
    def __init__(self, min_gap_seconds: float = 12.0):
        self.min_gap = min_gap_seconds
        self.last_call = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.min_gap:
                await asyncio.sleep(self.min_gap - elapsed)
            self.last_call = time.time()


limiter = GeminiRateLimiter()

"""按实际完成量估算耗时，断点复用的记录不计入新任务处理速度。"""

from datetime import datetime, timedelta, timezone
import time


class ProgressUpdate(str):
    def __new__(cls, message, detail):
        instance = super().__new__(cls, message)
        instance.detail = detail
        return instance


class ProgressTracker:
    def __init__(self, total, *, unit="段"):
        self.total = total
        self.unit = unit
        self.started = time.perf_counter()
        self.started_at = datetime.now(timezone.utc).isoformat()

    def update(self, message, *, current, reused=0, stage="处理中"):
        current = min(self.total, max(0, current))
        reused = min(current, max(0, reused))
        elapsed = max(0, time.perf_counter() - self.started)
        fresh = current - reused
        remaining = elapsed * (self.total - current) / fresh if fresh else None
        if current == self.total:
            remaining = 0
        now = datetime.now(timezone.utc)
        detail = {"current": current, "total": self.total, "reused": reused, "unit": self.unit,
                  "stage": stage, "percent": round(100 * current / self.total, 1) if self.total else 0,
                  "elapsed_seconds": round(elapsed, 1), "remaining_seconds": round(remaining, 1) if remaining is not None else None,
                  "started_at": self.started_at, "updated_at": now.isoformat(),
                  "eta_at": (now + timedelta(seconds=remaining)).isoformat() if remaining is not None else None}
        return ProgressUpdate(message, detail)

# observability.py - Minimal structured logging and per-request trace collection
import contextlib
import json
import time
import uuid
from contextvars import ContextVar


_trace_id: ContextVar[str | None] = ContextVar("_trace_id", default=None)
_events: ContextVar[list | None] = ContextVar("_events", default=None)


def new_trace() -> str:
    """Start a new trace for the current request. Returns the trace_id."""
    trace_id = uuid.uuid4().hex[:12]
    _trace_id.set(trace_id)
    _events.set([])
    return trace_id


def current_trace_id() -> str | None:
    return _trace_id.get()


def log_event(stage: str, **fields) -> None:
    """Append a structured event to the current trace."""
    events = _events.get()
    if events is None:
        return
    events.append({
        "trace_id": _trace_id.get(),
        "stage": stage,
        "ts": time.time(),
        **fields,
    })


@contextlib.contextmanager
def span(stage: str, **fields):
    """Time a block and emit a single structured event at its end."""
    start = time.perf_counter()
    err = None
    try:
        yield
    except BaseException as e:
        err = repr(e)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        extra = {"latency_ms": round(elapsed_ms, 1)}
        if err is not None:
            extra["error"] = err
        log_event(stage, **fields, **extra)


def trace_events() -> list[dict]:
    """Return the accumulated events for the current trace."""
    return list(_events.get() or [])


def print_trace() -> None:
    """Pretty-print the current trace, one event per line, JSON-encoded."""
    for event in trace_events():
        print(json.dumps(event))


def trace_summary() -> dict:
    """Aggregate the per-stage latencies into a flat summary."""
    summary: dict = {"trace_id": _trace_id.get(), "stages": {}}
    total_ms = 0.0
    for event in trace_events():
        stage = event["stage"]
        summary["stages"][stage] = {
            k: v for k, v in event.items()
            if k not in ("trace_id", "stage", "ts")
        }
        total_ms += event.get("latency_ms", 0)
    summary["total_latency_ms"] = round(total_ms, 1)
    return summary

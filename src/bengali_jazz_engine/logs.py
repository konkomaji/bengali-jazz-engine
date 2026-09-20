"""One log for everything: ``<workspace>/logs/engine.jsonl``.

Every run gets a ``run_id``; every line carries the run, song and stage it belongs to. Two things feed the file:

* structured events (``event("stage_end", stage="melody", seconds=31.2)``), warnings and errors with tracebacks;
* everything the stages print, teed line by line from stdout (``tee``), so the console experience is unchanged while
  the file keeps the full story of the run.

    bengali-jazz-engine logs runs            # one line per run: when, song, status, seconds
    bengali-jazz-engine logs show --run last --level WARNING --stage melody --tail 50
    bengali-jazz-engine logs clear --yes

The file rotates at 5 MB (3 backups). JSON lines, one object per line: ``ts, level, run_id, song, stage, logger, msg`` plus
any extra fields of an event.
"""
import contextlib
import io
import json
import logging
import logging.handlers
import re
import sys
import time
import uuid
from pathlib import Path

from . import config as cfg

ROOT_LOGGER = "bje"
MAX_BYTES = 5_000_000
BACKUPS = 3
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
_STATE = {"run_id": None, "song": None, "stage": None, "path": None}
_PROGRESS = re.compile(r"\d+%\||it/s|seconds/s|\[\d+:\d+")


def log_path() -> Path:
    return cfg.LOG_DIR / "engine.jsonl"


class JsonlHandler(logging.handlers.RotatingFileHandler):
    """Writes one JSON object per record with the run context and any structured fields."""

    def emit(self, record):
        try:
            row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)) + f".{int(record.msecs):03d}",
                   "level": record.levelname, "run_id": _STATE["run_id"], "song": _STATE["song"],
                   "stage": _STATE["stage"], "logger": record.name, "msg": record.getMessage()}
            row.update(getattr(record, "fields", {}) or {})
            if record.exc_info:
                row["traceback"] = "".join(traceback_lines(record.exc_info))
            self.stream = self.stream or self._open()
            self.stream.write(json.dumps(row, default=str) + "\n")
            self.flush()
            if self.shouldRollover(record):
                self.doRollover()
        except Exception:  # noqa: BLE001 - logging must never break a run
            self.handleError(record)


def traceback_lines(exc_info):
    import traceback

    return traceback.format_exception(*exc_info)


def setup():
    """Attach the JSONL handler for the current workspace (idempotent; re-attaches if the workspace moved)."""
    path = log_path()
    root = logging.getLogger(ROOT_LOGGER)
    if _STATE["path"] == path and root.handlers:
        return root
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    root.setLevel(logging.DEBUG)
    root.propagate = False
    root.addHandler(JsonlHandler(str(path), maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8"))
    _STATE["path"] = path
    return root


def get(name="engine") -> logging.Logger:
    setup()
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")


def new_run() -> str:
    """Start a new run id (timestamp + random suffix) and return it."""
    _STATE["run_id"] = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    _STATE["song"] = _STATE["stage"] = None
    return _STATE["run_id"]


def current_run() -> str | None:
    return _STATE["run_id"]


def context(song=..., stage=...):
    """Set the song and/or stage that following lines belong to (Ellipsis leaves a field unchanged)."""
    if song is not ...:
        _STATE["song"] = song
    if stage is not ...:
        _STATE["stage"] = stage


def event(name, level="INFO", **fields):
    """A structured event: msg is the event name, the keyword fields become columns."""
    get("events").log(getattr(logging, level.upper()), name, extra={"fields": {"event": name, **fields}})


def exception(msg, exc):
    """Log an error with its traceback (always written to the file, even when the console shows one line)."""
    get("engine").error(msg, exc_info=(type(exc), exc, exc.__traceback__))


class Tee(io.TextIOBase):
    """Passes text to `stream` unchanged and logs every completed line (progress bars are skipped)."""

    def __init__(self, stream, logger):
        self.stream, self.logger, self._buf = stream, logger, ""

    def write(self, text):
        self.stream.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._log(line)
        return len(text)

    def _log(self, line):
        line = line.split("\r")[-1].rstrip()
        if not line or _PROGRESS.search(line):
            return
        level = logging.WARNING if line.lstrip().upper().startswith(("WARNING", "WARN:")) else logging.INFO
        self.logger.log(level, line)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return bool(getattr(self.stream, "isatty", lambda: False)())


@contextlib.contextmanager
def tee(stream=None):
    """Route sys.stdout to `stream` (default: the current stdout) and mirror every printed line into the log."""
    stream = stream if stream is not None else sys.stdout
    old = sys.stdout
    sys.stdout = Tee(stream, get("console"))
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stdout = old


# ---- reading ---------------------------------------------------------------------------------

def read(run_id=None, level=None, stage=None, tail=None, path=None):
    """Log rows as dicts. run_id 'last' = the most recent run; level is a minimum; tail keeps the last N rows."""
    path = Path(path) if path else log_path()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    if run_id == "last":
        ids = [r["run_id"] for r in rows if r.get("run_id")]
        run_id = ids[-1] if ids else None
    if run_id:
        rows = [r for r in rows if r.get("run_id") == run_id]
    if level:
        floor = LEVELS.index(level.upper())
        rows = [r for r in rows if LEVELS.index(r.get("level", "INFO")) >= floor]
    if stage:
        rows = [r for r in rows if r.get("stage") == stage]
    return rows[-tail:] if tail else rows


def runs(path=None):
    """One summary per run: start time, songs, status (ok / failed / running), seconds, warnings and errors."""
    out = {}
    for r in read(path=path):
        rid = r.get("run_id")
        if not rid:
            continue
        s = out.setdefault(rid, {"run_id": rid, "start": r["ts"], "songs": [], "status": "running", "seconds": None,
                                 "warnings": 0, "errors": 0})
        if r.get("song") and r["song"] not in s["songs"]:
            s["songs"].append(r["song"])
        s["warnings"] += r["level"] == "WARNING"
        s["errors"] += r["level"] == "ERROR"
        if r.get("event") == "run_end":
            s["status"], s["seconds"] = r.get("status", "ok"), r.get("seconds")
    return list(out.values())


def format_row(r):
    where = "/".join(x for x in (r.get("song"), r.get("stage")) if x)
    tail = f"\n{r['traceback']}" if r.get("traceback") else ""
    return f"{r['ts']} {r['level']:<7} {where + ' ' if where else ''}{r['msg']}{tail}"


def shutdown():
    """Close the log file handle (tests and workspace switches)."""
    root = logging.getLogger(ROOT_LOGGER)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    _STATE.update(path=None, run_id=None, song=None, stage=None)


def clear(path=None):
    """Delete the log and its rotated backups; returns the removed paths."""
    path = Path(path) if path else log_path()
    for h in logging.getLogger(ROOT_LOGGER).handlers:
        h.close()
    logging.getLogger(ROOT_LOGGER).handlers.clear()
    _STATE["path"] = None
    removed = []
    for p in [path, *path.parent.glob(path.name + ".*")]:
        if p.exists():
            p.unlink()
            removed.append(str(p))
    return removed

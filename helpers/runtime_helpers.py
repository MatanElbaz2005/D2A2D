import os
import json
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, InvalidOperation
import math

def _truncate_decimal_str(x: Decimal, places: int = 20) -> str:
    q = Decimal(1).scaleb(-places)
    return str(x.quantize(q, rounding=ROUND_DOWN))

def _to_ms_str(seconds: float, places: int = 20) -> str | None:
    if seconds is None or not math.isfinite(seconds):
        return None
    try:
        return _truncate_decimal_str(Decimal(str(seconds)) * Decimal("1000"), places)
    except (InvalidOperation, ValueError):
        return None

def _rt_init(save_runtime, _RUNTIME, OS):
    if not save_runtime:
        return
    _R = _RUNTIME
    _R["enabled"] = True
    _R["collected"] = {}
    _R["flushed"] = False
    suffix = "windows" if OS.lower() == "windows" else "raspberry_pi"
    _R["filename"] = f"runtimes_{suffix}.json"
    _R["legacy_fn"] = f"runtimes_{suffix}.jsonl"

def _rt_set_frame(idx: int, _RUNTIME):
    _RUNTIME["frame"] = idx

def _rt_record(label: str, seconds: float, _RUNTIME):
    if not _RUNTIME["enabled"]:
        return
    if 2 <= _RUNTIME["frame"] <= 5:
        try:
            v = float(seconds)
        except (TypeError, ValueError):
            return
        if math.isfinite(v):
            _RUNTIME["collected"].setdefault(label, []).append(v)

def _rt_print(_RUNTIME, label: str, seconds: float, suffix: str = "", extra: str = ""):
    msg = f"{label}{seconds}{suffix}{extra}"
    print(msg)
    _rt_record(label.strip(), seconds, _RUNTIME)

def _rt_flush_if_ready(_RUNTIME, OS):
    if not _RUNTIME["enabled"]:
        return
    if _RUNTIME["flushed"]:
        return
    if _RUNTIME["frame"] < 5:
        return

    averages_sec = {
        k: (sum(v) / len(v) if v else None)
        for k, v in _RUNTIME["collected"].items()
    }

    averages_ms = {k: (_to_ms_str(v) if v is not None else None) for k, v in averages_sec.items()}

    averages_ms_float = {}
    for k, v in averages_sec.items():
        if v is None or not math.isfinite(v):
            averages_ms_float[k] = None
        else:
            try:
                averages_ms_float[k] = float(Decimal(str(v)) * Decimal("1000"))
            except (InvalidOperation, ValueError):
                averages_ms_float[k] = None

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    run_record = {
        "timestamp": now,
        "os": OS,
        "frames_used": [2, 3, 4, 5],
        "averages_ms": averages_ms,
        "averages_ms_float": averages_ms_float,
        "improvements_vs_prev": None
    }

    fn = _RUNTIME["filename"]
    legacy = _RUNTIME.get("legacy_fn")

    root = None
    if os.path.exists(fn):
        try:
            with open(fn, "r", encoding="utf-8") as f:
                root = json.load(f)
        except Exception:
            root = None

    if root is None:
        root = {"os": OS, "runs": []}

    prev_run = root["runs"][-1] if root["runs"] else None
    if prev_run:
        prev_avg = prev_run.get("averages_ms_float", {})
        improvements = {}
        for k, cur in averages_ms_float.items():
            pv = prev_avg.get(k)
            if cur is None or pv is None:
                continue
            if pv > 0 and math.isfinite(pv) and math.isfinite(cur):
                diff = cur - pv
                pct = (diff / pv) * 100.0
                diff_str = _truncate_decimal_str(Decimal(str(diff)))
                pct_str = _truncate_decimal_str(Decimal(str(pct)))
                improvements[k] = f"{diff_str} ms ({pct_str} %)"
        if improvements:
            run_record["improvements_vs_prev"] = improvements

    root["runs"].append(run_record)

    with open(fn, "w", encoding="utf-8") as f:
        json.dump(root, f, ensure_ascii=False, indent=2)

    _RUNTIME["flushed"] = True

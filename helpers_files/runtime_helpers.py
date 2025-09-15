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

def _rt_init(save_runtime, _RUNTIME, OS, sample_start: int = 2, sample_count: int = 4):
    if not save_runtime:
        return
    _R = _RUNTIME
    _R["enabled"] = True
    _R["collected"] = {}
    _R["flushed"] = False

    _R["sample_start"] = int(sample_start)
    _R["sample_count"] = int(sample_count)
    _R["frames_used"] = 0

    suffix = "windows" if OS.lower() == "windows" else "raspberry_pi"
    project_root = os.path.dirname(os.path.dirname(__file__))
    base_dir = os.path.join(project_root, "json_info")
    os.makedirs(base_dir, exist_ok=True)

    _R["filename"] = os.path.join(base_dir, f"runtimes_{suffix}.json")
    _R["legacy_fn"] = os.path.join(base_dir, f"runtimes_{suffix}.jsonl")

def _rt_set_frame(idx: int, _RUNTIME):
    _RUNTIME["frame"] = idx
    _RUNTIME["_marked_this_frame"] = False

def _rt_record(label: str, seconds: float, _RUNTIME):
    if not isinstance(_RUNTIME, dict):
        return
    if not _RUNTIME.get("enabled"):
        return

    frame_idx = _RUNTIME.get("frame", 0)
    start = int(_RUNTIME.get("sample_start", 2))
    count = int(_RUNTIME.get("sample_count", 4))
    end   = start + count - 1

    if not (start <= frame_idx <= end):
        return

    try:
        v = float(seconds)
    except (TypeError, ValueError):
        return
    if math.isfinite(v):
        _RUNTIME["collected"].setdefault(label, []).append(v)

        if not _RUNTIME.get("_marked_this_frame", False):
            _RUNTIME["frames_used"] = int(_RUNTIME.get("frames_used", 0)) + 1
            _RUNTIME["_marked_this_frame"] = True

def _rt_print(_RUNTIME, label: str, seconds: float, suffix: str = "", extra: str = ""):
    msg = f"{label}{seconds}{suffix}{extra}"
    print(msg)
    if isinstance(_RUNTIME, dict):
        _rt_record(label.strip(), seconds, _RUNTIME)

def _rt_flush_if_ready(_RUNTIME, OS):
    if not _RUNTIME.get("enabled"):
        return
    if _RUNTIME.get("flushed"):
        return

    frame_idx = _RUNTIME.get("frame", 0)
    start = int(_RUNTIME.get("sample_start", 2))
    count = int(_RUNTIME.get("sample_count", 4))
    end   = start + count - 1

    if frame_idx < end:
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    frames_used_count = int(_RUNTIME.get("frames_used", 0))

    fn = _RUNTIME["filename"]
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

    HIGHLIGHT_KEYS = ("[ENC] encode_udp_to_frame (outer):", "[DEC] decode_frame_to_udp (outer):")

    averages_sec = {
        k: (sum(v) / len(v) if v else None)
        for k, v in _RUNTIME["collected"].items()
    }
    averages_ms = {}
    for k, v in averages_sec.items():
        if v is None or not math.isfinite(v):
            averages_ms[k] = None
        else:
            try:
                averages_ms[k] = float(Decimal(str(v)) * Decimal("1000"))
            except (InvalidOperation, ValueError):
                averages_ms[k] = None

    def _bucket_of(key: str) -> str:
        if key.startswith("[GUI]"):  return "GUI"
        if key.startswith("[ENC]"):  return "ENC"
        if key.startswith("[DEC]"):  return "DEC"
        if key.startswith("[LOOP]"): return "LOOP"
        return "OTHER"

    sections = {"GUI": {}, "ENC": {}, "DEC": {}, "LOOP": {}, "OTHER": {}}
    for k, ms in averages_ms.items():
        bucket = _bucket_of(k)
        sections[bucket][k] = ms

    def _sum_valid(d: dict[str, float | None], exclude: set[str] | None = None) -> float:
        total = 0.0
        if not d:
            return 0.0
        exclude = exclude or set()
        for k, v in d.items():
            if k in exclude:
                continue
            if isinstance(v, (int, float)) and math.isfinite(v):
                total += float(v)
        return total

    ENC_OUTER_KEY = "[ENC] encode_udp_to_frame (outer):"
    DEC_OUTER_KEY = "[DEC] decode_frame_to_udp (outer):"
    LOOP_EXCLUDE = {"[LOOP] frame total:"}

    section_totals_ms = {
        "GUI": _sum_valid(sections.get("GUI", {})),
        "ENC": float(sections.get("ENC", {}).get(ENC_OUTER_KEY, 0.0) or 0.0),
        "DEC": float(sections.get("DEC", {}).get(DEC_OUTER_KEY, 0.0) or 0.0),
        "LOOP": _sum_valid(sections.get("LOOP", {}), exclude=LOOP_EXCLUDE),
        "OTHER": _sum_valid(sections.get("OTHER", {})),
    }

    highlights_block = []
    for key in HIGHLIGHT_KEYS:
        sect = "ENC" if key.startswith("[ENC]") else "DEC"
        val = sections.get(sect, {}).get(key)
        if isinstance(val, (int, float)) and math.isfinite(val):
            highlights_block.append("--------------------------------")
            highlights_block.append(f"{key} {val} ms")
            highlights_block.append("--------------------------------")

    def _format_delta(cur: float | None, prev: float | None) -> str | None:
        if cur is None or prev is None or not (isinstance(cur, (int, float)) and isinstance(prev, (int, float))):
            return None
        if not (math.isfinite(cur) and math.isfinite(prev)) or prev == 0:
            return f"{cur - prev:.8f} ms (n/a)"
        diff = cur - prev
        pct = (diff / prev) * 100.0
        return f"{diff:.8f} ms ({pct:.8f} %)"

    improvements_vs_prev = None
    if prev_run is not None and isinstance(prev_run, dict):
        improvements_vs_prev = {"GUI": {}, "ENC": {}, "DEC": {}, "LOOP": {}, "OTHER": {}}
        prev_sections = prev_run.get("sections", {})
        for bucket in improvements_vs_prev.keys():
            cur_map = sections.get(bucket, {})
            prev_map = prev_sections.get(bucket, {})
            out_map = {}
            for k, cur_val in cur_map.items():
                prev_val = prev_map.get(k)
                delta_str = _format_delta(cur_val, prev_val)
                if delta_str is not None:
                    out_map[k] = delta_str
            improvements_vs_prev[bucket] = out_map

    run_record = {
        "timestamp": now,
        "os": OS,
        "frames_used": frames_used_count,
        "frames_window": {
            "start": start,
            "count": count
        },
        "sections": sections,
        "section_totals_ms": section_totals_ms,
        "highlights_block": highlights_block,
        "improvements_vs_prev": improvements_vs_prev
    }

    root["runs"].append(run_record)
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(root, f, ensure_ascii=False, indent=2)

    _RUNTIME["flushed"] = True

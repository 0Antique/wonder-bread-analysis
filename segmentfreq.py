#!/usr/bin/env python3
"""
Frequency-aware segmentation for WonderBread-like traces.

Keeps the original hard boundaries based on URL/TAB changes, and adds additional
segment boundaries using high-frequency "anchor" actions learned per task from
`frequency/{task_id}_traj_highfreq.csv`.

Input:  a single trace json file (contains {"trace":[{type:"state"/"action",data:{...}}, ...]}).
Output: *_segments_freq.csv and *_actions_debug_freq.csv in --out_dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


MISSING_TEXT = "(missing)"


def normalize_element_text(text: Any) -> str:
    if text is None:
        return MISSING_TEXT
    if not isinstance(text, str):
        return str(text)
    # Keep the same convention as rawdata_analysis.ipynb (escape newlines for stable matching).
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "\\n")
        .strip()
        or MISSING_TEXT
    )


def parse_json_state(js: Any) -> List[Dict[str, Any]]:
    try:
        if not js:
            return []
        if isinstance(js, list):
            return js
        return json.loads(js)
    except Exception:
        return []


def url_path(u: str) -> str:
    try:
        p = urlparse(u)
        return f"{p.scheme}://{p.netloc}{p.path}"
    except Exception:
        return u


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 1.0


def extract_triples(
    trace: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], int]]:
    """Return list of (pre_state, action, post_state, trace_action_idx) aligned triples."""
    triples = []
    i = 0
    while i < len(trace):
        if trace[i].get("type") != "state":
            i += 1
            continue
        pre = trace[i]["data"]
        if i + 1 < len(trace) and trace[i + 1].get("type") == "action":
            act = trace[i + 1]["data"]
            act_trace_idx = i + 1
            j = i + 2
            while j < len(trace) and trace[j].get("type") != "state":
                j += 1
            post = trace[j]["data"] if j < len(trace) else None
            triples.append((pre, act, post, act_trace_idx))
            i = j
        else:
            i += 1
    return triples


def state_meta(state_data: Dict[str, Any], *, max_elems: int = 0) -> Dict[str, Any]:
    """
    Return url/tab and (optional) xp_set for DOM similarity.

    Set max_elems<=0 to skip DOM parsing.
    """
    url = state_data.get("url", "") or ""
    tab = state_data.get("tab", "") or ""
    xp_set: set = set()
    if max_elems and max_elems > 0:
        elems = parse_json_state(state_data.get("json_state"))[:max_elems]
        for e in elems:
            xp = e.get("xpath")
            if xp:
                xp_set.add(xp)
    return {"url": url, "tab": tab, "xp_set": xp_set}


def action_key(action_data: Dict[str, Any]) -> Tuple[str, str]:
    a_type = action_data.get("type", "") or ""
    el = (action_data.get("element_attributes") or {}).get("element") or {}
    text_norm = normalize_element_text(el.get("text"))
    return (a_type, text_norm)


def infer_task_id(trace_json_path: Path) -> Optional[str]:
    """
    Try to infer task_id from parent folder or file name: "{task_id} @ {timestamp}.json".
    """
    candidates = [trace_json_path.parent.name, trace_json_path.stem, trace_json_path.name]
    for name in candidates:
        m = re.match(r"^(\d+)\s*@", name)
        if m:
            return m.group(1)
    return None


@dataclass(frozen=True)
class Anchor:
    action_type: str
    element_text: str
    order_rank: int
    hits: int
    trace_files: int
    median_order: float
    min_order: int
    max_order: int


def load_stage_anchors(
    freq_csv: Path,
    *,
    min_trace_files: Optional[int] = None,
    include_missing_anchors: bool = False,
    include_keystroke_anchors: bool = False,
    anchor_order_margin: int = 5,
) -> Dict[Tuple[str, str], Anchor]:
    required = [
        "action_type",
        "element_text",
        "hits",
        "trace_files",
        "median_order",
        "min_order",
        "max_order",
        "order_rank",
    ]
    rows: List[Dict[str, str]] = []
    with freq_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Frequency CSV has no header: {freq_csv}")
        missing = set(required) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Frequency CSV is missing required columns: {sorted(missing)}")
        for r in reader:
            rows.append(r)

    parsed: List[Anchor] = []
    for r in rows:
        a_type = (r.get("action_type") or "").strip()
        e_text = normalize_element_text(r.get("element_text"))
        try:
            hits = int(float(r.get("hits") or 0))
        except Exception:
            hits = 0
        try:
            trace_files = int(float(r.get("trace_files") or 0))
        except Exception:
            trace_files = 0
        try:
            median_order = float(r.get("median_order") or 0)
        except Exception:
            median_order = 0.0
        try:
            min_order = int(float(r.get("min_order") or 0))
        except Exception:
            min_order = 0
        try:
            max_order = int(float(r.get("max_order") or 0))
        except Exception:
            max_order = 0
        try:
            order_rank = int(float(r.get("order_rank") or 0))
        except Exception:
            order_rank = 0
        parsed.append(
            Anchor(
                action_type=a_type,
                element_text=e_text,
                order_rank=order_rank,
                hits=hits,
                trace_files=trace_files,
                median_order=median_order,
                min_order=min_order,
                max_order=max_order,
            )
        )

    if not parsed:
        return {}

    if min_trace_files is None:
        min_trace_files = max(a.trace_files for a in parsed)

    anchors: Dict[Tuple[str, str], Anchor] = {}
    for a in parsed:
        if a.trace_files < min_trace_files:
            continue
        if not include_keystroke_anchors and a.action_type == "keystroke":
            continue
        if not include_missing_anchors and a.element_text == MISSING_TEXT:
            continue
        anchors[(a.action_type, a.element_text)] = a

    # Embed an order-consistency guard by expanding expected ranges used during matching.
    # (We keep margin here so callers don't need to know task-specific drift.)
    if anchor_order_margin > 0:
        adjusted: Dict[Tuple[str, str], Anchor] = {}
        for k, a in anchors.items():
            adjusted[k] = Anchor(
                action_type=a.action_type,
                element_text=a.element_text,
                order_rank=a.order_rank,
                hits=a.hits,
                trace_files=a.trace_files,
                median_order=a.median_order,
                min_order=max(0, a.min_order - anchor_order_margin),
                max_order=a.max_order + anchor_order_margin,
            )
        anchors = adjusted

    return anchors


def segment_trace_with_freq(
    trace: List[Dict[str, Any]],
    anchors: Dict[Tuple[str, str], Anchor],
    *,
    dom_jaccard_threshold: Optional[float] = None,
    max_dom_elems: int = 400,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (segments, debug_rows).

    Segmentation rule:
    - Always cut on URL_CHANGE / TAB_CHANGE (kept from segment.py).
    - Additionally cut on stage anchors (from high-frequency CSV), even if URL doesn't change.
    - Optionally cut on DOM jaccard drop if dom_jaccard_threshold is provided.
    """
    triples = extract_triples(trace)

    boundaries: Dict[int, List[str]] = {}  # action_idx -> [reasons]
    debug_rows: List[Dict[str, Any]] = []

    for idx, (pre, act, post, act_trace_idx) in enumerate(triples):
        if post is None:
            continue

        need_dom = dom_jaccard_threshold is not None
        pre_sig = state_meta(pre, max_elems=max_dom_elems if need_dom else 0)
        post_sig = state_meta(post, max_elems=max_dom_elems if need_dom else 0)

        pre_path = url_path(pre_sig["url"])
        post_path = url_path(post_sig["url"])
        url_changed = pre_path != post_path
        tab_changed = bool(pre_sig["tab"]) and bool(post_sig["tab"]) and pre_sig["tab"] != post_sig["tab"]

        key = action_key(act)
        anchor = anchors.get(key)
        is_anchor = anchor is not None

        reasons: List[str] = []
        if url_changed:
            reasons.append("URL_CHANGE")
        if tab_changed:
            reasons.append("TAB_CHANGE")

        jac = None
        if need_dom:
            jac = jaccard(pre_sig["xp_set"], post_sig["xp_set"])
            if jac < float(dom_jaccard_threshold):
                reasons.append(f"DOM_JACCARD<{float(dom_jaccard_threshold):.2f}({jac:.2f})")

        # Anchor boundary: keep only if the action's trace_action_idx is within the anchor's expected range.
        if is_anchor and anchor is not None:
            if anchor.min_order <= act_trace_idx <= anchor.max_order:
                reasons.append(f"FREQ_ANCHOR:{anchor.order_rank}:{anchor.element_text}")
            else:
                # Still keep anchor info in debug, but don't cut on out-of-range matches.
                pass

        if reasons:
            boundaries.setdefault(idx, []).extend(reasons)

        # Debug row
        el = (act.get("element_attributes") or {}).get("element") or {}
        debug_rows.append(
            {
                "action_idx": idx,
                "trace_action_idx": act_trace_idx,
                "action_type": act.get("type", ""),
                "element_tag": el.get("tag"),
                "element_role": el.get("role"),
                "element_xpath": el.get("xpath"),
                "element_text_norm": key[1],
                "anchor_rank": anchor.order_rank if anchor else None,
                "anchor_trace_files": anchor.trace_files if anchor else None,
                "anchor_hits": anchor.hits if anchor else None,
                "pre_url": pre_sig["url"],
                "post_url": post_sig["url"],
                "url_changed": url_changed,
                "tab_changed": tab_changed,
                "dom_jaccard": jac,
            }
        )

    # Build segments
    segments: List[Dict[str, Any]] = []
    start = 0
    seg_id = 0
    for idx in range(len(triples)):
        if idx in boundaries:
            end = idx
            pre_state = triples[start][0]
            post_state = triples[end][2] or triples[end][0]
            start_trace_action_idx = triples[start][3]
            end_trace_action_idx = triples[end][3]
            segments.append(
                {
                    "segment_id": seg_id,
                    "start_action_idx": start,
                    "start_trace_action_idx": start_trace_action_idx,
                    "end_action_idx": end,
                    "end_trace_action_idx": end_trace_action_idx,
                    "start_url": pre_state.get("url", ""),
                    "end_url": post_state.get("url", ""),
                    "reason": ";".join(sorted(set(boundaries[idx]))),
                }
            )
            seg_id += 1
            start = idx + 1

    if start < len(triples):
        pre_state = triples[start][0]
        post_state = triples[-1][2] or triples[-1][0]
        start_trace_action_idx = triples[start][3]
        end_trace_action_idx = triples[-1][3]
        segments.append(
            {
                "segment_id": seg_id,
                "start_action_idx": start,
                "start_trace_action_idx": start_trace_action_idx,
                "end_action_idx": len(triples) - 1,
                "end_trace_action_idx": end_trace_action_idx,
                "start_url": pre_state.get("url", ""),
                "end_url": post_state.get("url", ""),
                "reason": "END",
            }
        )

    return segments, debug_rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "trace_json",
        help="Path to a trace json file, a trajectory folder (contains *.json), or the base folder (contains many trajectory folders).",
    )
    ap.add_argument("--out_dir", default="output", help="Output directory")
    ap.add_argument("--freq_dir", default="frequency", help="Directory containing *_traj_highfreq.csv")
    ap.add_argument("--freq_csv", default=None, help="Override: path to a specific *_traj_highfreq.csv")
    ap.add_argument(
        "--min_trace_files",
        type=int,
        default=None,
        help="Minimum trace_files required for an anchor to be used (default: max(trace_files) in the freq CSV).",
    )
    ap.add_argument("--include_missing_anchors", action="store_true", help="Allow anchors with element_text == '(missing)'")
    ap.add_argument("--include_keystroke_anchors", action="store_true", help="Allow keystroke anchors")
    ap.add_argument(
        "--anchor_order_margin",
        type=int,
        default=5,
        help="Allowed drift from anchor min/max order when matching (default: 5).",
    )
    ap.add_argument(
        "--dom_jaccard_threshold",
        type=float,
        default=None,
        help="Optional: also cut when DOM jaccard drops below this value (disabled by default).",
    )
    args = ap.parse_args()

    trace_input = Path(args.trace_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files: List[Path]
    if trace_input.is_dir():
        # Case 1: trajectory folder (directly contains *.json)
        json_files = sorted(trace_input.glob("*.json"))
        # Case 2: base folder (contains many trajectory folders)
        if not json_files:
            for folder in sorted(p for p in trace_input.iterdir() if p.is_dir()):
                json_files.extend(sorted(folder.glob("*.json")))
        if not json_files:
            raise FileNotFoundError(f"No *.json files found in: {trace_input}")
    else:
        json_files = [trace_input]

    anchor_cache: Dict[Path, Dict[Tuple[str, str], Anchor]] = {}
    missing_freq: Dict[str, int] = {}
    processed = 0

    for trace_json_path in json_files:
        if args.freq_csv:
            freq_csv = Path(args.freq_csv)
            if not freq_csv.exists():
                raise FileNotFoundError(f"Frequency CSV not found: {freq_csv}")
        else:
            task_id = infer_task_id(trace_json_path)
            if not task_id:
                raise ValueError(
                    "Could not infer task_id from path. Provide --freq_csv explicitly or use a path like '0 @ .../*.json'."
                )
            freq_csv = Path(args.freq_dir) / f"{task_id}_traj_highfreq.csv"

        anchors: Dict[Tuple[str, str], Anchor]
        if freq_csv.exists():
            if freq_csv not in anchor_cache:
                anchor_cache[freq_csv] = load_stage_anchors(
                    freq_csv,
                    min_trace_files=args.min_trace_files,
                    include_missing_anchors=args.include_missing_anchors,
                    include_keystroke_anchors=args.include_keystroke_anchors,
                    anchor_order_margin=args.anchor_order_margin,
                )
            anchors = anchor_cache[freq_csv]
        else:
            # Only possible when we infer by task_id; if user explicitly passed --freq_csv we error above.
            missing_freq[task_id] = missing_freq.get(task_id, 0) + 1
            anchors = {}

        obj = json.loads(trace_json_path.read_text(encoding="utf-8"))
        trace = obj.get("trace", [])

        segments, debug_rows = segment_trace_with_freq(
            trace,
            anchors,
            dom_jaccard_threshold=args.dom_jaccard_threshold,
        )

        stem = trace_json_path.stem.replace(" @ ", "_")
        segments_path = out_dir / f"{stem}_segments_freq.csv"
        debug_path = out_dir / f"{stem}_actions_debug_freq.csv"
        write_csv(segments_path, segments)
        write_csv(debug_path, debug_rows)

        processed += 1
        freq_name = freq_csv.name if freq_csv.exists() else "(missing freq)"
        print(f"[{trace_json_path.name}] anchors={len(anchors)} (from {freq_name}) segments={len(segments)}")
        print(f"  -> {segments_path}")
        print(f"  -> {debug_path}")

    if missing_freq:
        missing_total = sum(missing_freq.values())
        missing_tasks = ", ".join(sorted(missing_freq.keys(), key=lambda x: int(x) if x.isdigit() else x)[:20])
        more = "" if len(missing_freq) <= 20 else f" (+{len(missing_freq) - 20} more)"
        print(f"\nWARNING: missing frequency CSV for {missing_total}/{processed} trajectories. task_ids: {missing_tasks}{more}")


if __name__ == "__main__":
    main()

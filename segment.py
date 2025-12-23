#!/usr/bin/env python3

"""
在output文件夹下面生成segmentation_out文件夹，里面存放每个trace的segments.csv和actions_debug.csv
按照人为设定的规则对trace进行分段
url/tab的效果是显而易见的

Strong-signal segmentation for WonderBread-like traces.

Input: trace json files with "trace": [ {type:"state",data:{...}}, {type:"action",data:{...}}, ... ]
Output: segments.csv with segment boundaries and reasons; actions_debug.csv for inspection.
"""
import json, re
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd

SUBMIT_WORDS = re.compile(r"\b(submit|search|apply|save|continue|next|go|show|run|login|sign in|add to cart|checkout)\b", re.I)
DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")

def parse_json_state(js):
    try:
        return json.loads(js)
    except Exception:
        return []

def state_signature(state_data, max_elems=400):
    url = state_data.get("url","")
    tab = state_data.get("tab","")
    js = state_data.get("json_state","")
    elems = parse_json_state(js)[:max_elems] if js else []
    xps = []
    for e in elems:
        xp = e.get("xpath")
        if xp:
            xps.append(xp)
    xp_set = set(xps)
    return {"url": url, "tab": tab, "xp_set": xp_set}

def url_path(u):
    try:
        p = urlparse(u)
        return p.scheme + "://" + p.netloc + p.path
    except Exception:
        return u

def jaccard(a, b):
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 1.0

def action_features(action_data):
    a_type = action_data.get("type","")
    el = (action_data.get("element_attributes") or {}).get("element") or {}
    tag = el.get("tag")
    role = el.get("role")
    text = (el.get("text") or "").strip()
    xpath = el.get("xpath")
    key = action_data.get("key") if a_type == "keystroke" else None

    is_submit_like = False
    if a_type == "mouseup":
        if (tag in {"button","a"} or role in {"button","link"}) and text and SUBMIT_WORDS.search(text):
            is_submit_like = True
        if el.get("type") in {"submit","button"} and text and SUBMIT_WORDS.search(text):
            is_submit_like = True

    return {"a_type": a_type, "tag": tag, "role": role, "text": text, "xpath": xpath, "key": key, "is_submit_like": is_submit_like}

def extract_triples(trace):
    """Return list of (pre_state, action, post_state) aligned triples."""
    triples = []
    i = 0
    while i < len(trace):
        if trace[i].get("type") != "state":
            i += 1
            continue
        pre = trace[i]["data"]
        if i+1 < len(trace) and trace[i+1].get("type") == "action":
            act = trace[i+1]["data"]
            j = i+2
            while j < len(trace) and trace[j].get("type") != "state":
                j += 1
            post = trace[j]["data"] if j < len(trace) else None
            triples.append((pre, act, post))
            i = j
        else:
            i += 1
    return triples

def segment_trace(trace, jac_threshold=1):
    triples = extract_triples(trace)
    boundaries = {}  # action_idx -> reason
    for idx, (pre, act, post) in enumerate(triples):
        if post is None:
            continue
        pre_sig = state_signature(pre)
        post_sig = state_signature(post)
        pre_path = url_path(pre_sig["url"])
        post_path = url_path(post_sig["url"])
        jac = jaccard(pre_sig["xp_set"], post_sig["xp_set"])
        af = action_features(act)

        reasons = []
        if pre_path != post_path:
            reasons.append("URL_CHANGE")
        if jac < jac_threshold:
            reasons.append(f"DOM_JACCARD<{jac_threshold:.2f}({jac:.2f})")
        if pre_sig["tab"] and post_sig["tab"] and pre_sig["tab"] != post_sig["tab"]:
            reasons.append("TAB_CHANGE")
        if af["a_type"] == "mouseup" and af["is_submit_like"]:
            reasons.append("SUBMIT_LIKE_CLICK")

        if reasons:
            boundaries[idx] = ";".join(reasons)

    segments = []
    start = 0
    seg_id = 0
    for idx in range(len(triples)):
        if idx in boundaries:
            end = idx
            pre_state = triples[start][0]
            post_state = triples[end][2] or triples[end][0]
            segments.append({
                "segment_id": seg_id,
                "start_action_idx": start,
                "end_action_idx": end,
                "start_url": pre_state.get("url",""),
                "end_url": post_state.get("url",""),
                "reason": boundaries[idx],
            })
            seg_id += 1
            start = idx + 1

    if start < len(triples):
        pre_state = triples[start][0]
        post_state = triples[-1][2] or triples[-1][0]
        segments.append({
            "segment_id": seg_id,
            "start_action_idx": start,
            "end_action_idx": len(triples)-1,
            "start_url": pre_state.get("url",""),
            "end_url": post_state.get("url",""),
            "reason": "END",
        })

    return segments, triples

def main(in_path: str, out_dir: str):
    in_path = Path(in_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = json.loads(in_path.read_text(encoding="utf-8"))
    trace = obj["trace"]
    segs, triples = segment_trace(trace)

    seg_df = pd.DataFrame(segs)
    seg_df.to_csv(out_dir / (in_path.stem.replace(" @ ", "_") + "_segments.csv"), index=False)

    # debug per action
    rows = []
    for i, (pre, act, post) in enumerate(triples):
        af = action_features(act)
        pre_sig = state_signature(pre)
        post_sig = state_signature(post) if post else {"url":"","tab":"","xp_set":set()}
        jac = jaccard(pre_sig["xp_set"], post_sig["xp_set"]) if post else None
        rows.append({
            "action_idx": i,
            "action_type": af["a_type"],
            "target_tag": af["tag"],
            "target_role": af["role"],
            "target_text": (af["text"] or "")[:120],
            "target_xpath": af["xpath"],
            "key": af["key"],
            "pre_url": pre_sig["url"],
            "post_url": post_sig["url"],
            "url_changed": url_path(pre_sig["url"]) != url_path(post_sig["url"]),
            "tab_changed": pre_sig["tab"] != post_sig["tab"],
            "dom_jaccard": jac,
            "is_submit_like": af["is_submit_like"],
        })
    pd.DataFrame(rows).to_csv(out_dir / (in_path.stem.replace(" @ ", "_") + "_actions_debug.csv"), index=False)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_json", help="Path to a trace json file")
    ap.add_argument("--out_dir", default="segmentation_out", help="Output directory")
    args = ap.parse_args()
    main(args.trace_json, args.out_dir)

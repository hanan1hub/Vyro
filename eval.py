#!/usr/bin/env python3
"""
eval.py — Evaluate Pocket-Agent against a JSONL test set.

Supports two test-set formats:

  Format A (grader format):
    {"prompt": "...", "history": [...], "expected": "..."}

  Format B (training data format):
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

  expected is either:
    - A tool call string: '<tool_call>{"tool":...}</tool_call>'
    - "REFUSAL" (any plain-text response with no <tool_call> tags)

Scoring (matches grader rubric exactly)
----------------------------------------
  +1.0  exact tool match, all args correct (numerical ±1%)
  +0.5  correct tool, ≥1 arg wrong
   0.0  wrong tool, malformed JSON, wrong refusal decision
  -0.5  emitted tool call when refusal was correct

Run: python eval.py [--test_file path/to/test.jsonl]
"""

import json, re, sys, argparse, time
from pathlib import Path

# ── Import model interface ─────────────────────────────────────────────
from inference import run


# ── Format conversion ─────────────────────────────────────────────────

def extract_prompt_and_expected(ex: dict):
    """
    Accept either:
      - {"prompt": "...", "history": [...], "expected": "..."}
      - {"messages": [system, user, assistant, ...]}
    Returns (prompt, history, expected) or None to skip.
    """
    # Format A — already has prompt/expected keys
    if "prompt" in ex and "expected" in ex:
        return ex["prompt"], ex.get("history", []), ex["expected"]

    # Format B — messages array (training data format)
    if "messages" in ex:
        messages = ex["messages"]
        # Extract user prompt (last user message)
        user_msgs = [m for m in messages if m["role"] == "user"]
        asst_msgs = [m for m in messages if m["role"] == "assistant"]

        if not user_msgs or not asst_msgs:
            return None  # skip malformed

        prompt   = user_msgs[-1]["content"]
        expected = asst_msgs[-1]["content"]

        # Build history = everything before the last user message
        last_user_idx = max(i for i, m in enumerate(messages) if m["role"] == "user")
        history = [
            m for m in messages[:last_user_idx]
            if m["role"] in ("user", "assistant")
        ]

        return prompt, history, expected

    return None  # unknown format, skip


# ── Scoring helpers ───────────────────────────────────────────────────

def parse_tool_call(text: str):
    """Return (tool_name, args) or None if no valid tool call."""
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1).strip())
        if isinstance(obj, dict) and "tool" in obj:
            return obj["tool"], obj.get("args", {})
    except json.JSONDecodeError:
        pass
    return None


def _nums_close(a, b, tol=0.01) -> bool:
    """True if a and b are within ±1% of each other."""
    try:
        fa, fb = float(a), float(b)
        if fb == 0:
            return fa == 0
        return abs(fa - fb) / abs(fb) <= tol
    except (TypeError, ValueError):
        return False


def _arg_equal(pred_val, gold_val) -> bool:
    """Compare a single arg value with tolerance for numerics."""
    if isinstance(gold_val, (int, float)):
        return _nums_close(pred_val, gold_val)
    return str(pred_val).strip().lower() == str(gold_val).strip().lower()


def score_example(pred: str, expected: str) -> tuple[float, str]:
    """
    Returns (score, reason).
    expected is either a tool-call string or "REFUSAL".
    """
    is_refusal_expected = (expected.strip().upper() == "REFUSAL")
    pred_parsed = parse_tool_call(pred)
    exp_parsed  = parse_tool_call(expected) if not is_refusal_expected else None

    # ── Case 1: refusal expected ──────────────────────────────────────
    if is_refusal_expected:
        if pred_parsed is None:
            return 1.0, "✅ Correct refusal"
        else:
            return -0.5, "❌ Emitted tool call when refusal was correct"

    # ── Case 2: tool call expected ────────────────────────────────────
    if exp_parsed is None:
        # Expected string is malformed — skip
        return 0.0, "⚠️  Expected string is malformed (test data issue)"

    gold_tool, gold_args = exp_parsed

    if pred_parsed is None:
        return 0.0, f"❌ Refusal — expected tool '{gold_tool}'"

    pred_tool, pred_args = pred_parsed

    if pred_tool != gold_tool:
        return 0.0, f"❌ Wrong tool: got '{pred_tool}', expected '{gold_tool}'"

    # Correct tool — check args
    all_correct = True
    mismatches  = []
    for key, gold_val in gold_args.items():
        pred_val = pred_args.get(key)
        if pred_val is None or not _arg_equal(pred_val, gold_val):
            all_correct = False
            mismatches.append(f"{key}: got={pred_val!r} expected={gold_val!r}")

    if all_correct:
        return 1.0, f"✅ Perfect: {gold_tool}"
    else:
        return 0.5, f"⚠️  Correct tool '{gold_tool}', wrong args: {'; '.join(mismatches)}"


# ── Main evaluation loop ──────────────────────────────────────────────

def evaluate(test_file: str):
    test_path = Path(test_file)
    if not test_path.exists():
        print(f"❌  Test file not found: {test_file}")
        sys.exit(1)

    raw_examples = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_examples.append(json.loads(line))

    # Convert to unified format
    examples = []
    skipped  = 0
    for ex in raw_examples:
        parsed = extract_prompt_and_expected(ex)
        if parsed is None:
            skipped += 1
            continue
        examples.append(parsed)

    if skipped:
        print(f"⚠️  Skipped {skipped} malformed examples")

    print(f"📋  Evaluating {len(examples)} examples from {test_file}\n")
    print(f"{'#':>3}  {'Score':>6}  {'Latency':>8}  Reason")
    print("─" * 70)

    total_score = 0.0
    latencies   = []
    results     = []

    for i, (prompt, history, expected) in enumerate(examples):
        t0         = time.perf_counter()
        pred       = run(prompt, history)
        t1         = time.perf_counter()
        latency_ms = (t1 - t0) * 1000

        sc, reason = score_example(pred, expected)
        total_score += sc
        latencies.append(latency_ms)
        results.append({
            "i": i + 1, "score": sc, "latency_ms": round(latency_ms, 1),
            "prompt": prompt, "predicted": pred, "expected": expected,
            "reason": reason,
        })

        print(f"{i+1:>3}  {sc:>+6.1f}  {latency_ms:>6.0f} ms  {reason}")

    # ── Summary ───────────────────────────────────────────────────────
    n          = len(examples)
    mean_score = total_score / n if n else 0
    mean_lat   = sum(latencies) / n if n else 0
    p95_lat    = sorted(latencies)[int(0.95 * n)] if n else 0

    print("\n" + "═" * 70)
    print(f"  Total score  : {total_score:.1f} / {n:.0f}  ({mean_score*100:.1f}%)")
    print(f"  Mean latency : {mean_lat:.0f} ms  (p95: {p95_lat:.0f} ms)")
    print(f"  Gate check   : latency ≤ 200 ms → {'✅ PASS' if mean_lat <= 200 else '❌ FAIL'}")
    print("═" * 70)

    # Save detailed results
    out = "eval_results.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  Detailed results saved → {out}")


def build_dev_test():
    """Build a quick sanity-check test file if none exists."""
    dev_path = "data/dev_test.jsonl"
    if Path(dev_path).exists():
        return dev_path

    samples = [
        {"prompt": "What's the weather in Tokyo?",   "history": [],
         "expected": '<tool_call>{"tool":"weather","args":{"location":"Tokyo","unit":"C"}}</tool_call>'},
        {"prompt": "Convert 100 USD to EUR",          "history": [],
         "expected": '<tool_call>{"tool":"currency","args":{"amount":100,"from":"USD","to":"EUR"}}</tool_call>'},
        {"prompt": "5 miles to kilometers",           "history": [],
         "expected": '<tool_call>{"tool":"convert","args":{"value":5,"from_unit":"miles","to_unit":"kilometers"}}</tool_call>'},
        {"prompt": "Add Meeting on 2025-05-01",        "history": [],
         "expected": '<tool_call>{"tool":"calendar","args":{"action":"create","date":"2025-05-01","title":"Meeting"}}</tool_call>'},
        {"prompt": "SELECT * FROM users",             "history": [],
         "expected": '<tool_call>{"tool":"sql","args":{"query":"SELECT * FROM users"}}</tool_call>'},
        {"prompt": "How are you?",                    "history": [], "expected": "REFUSAL"},
        {"prompt": "Send an email",                   "history": [], "expected": "REFUSAL"},
        {"prompt": "Convert that to euros",           "history": [], "expected": "REFUSAL"},
        {"prompt": "And in GBP?",
         "history": [
             {"role": "user",      "content": "Convert 200 EUR to USD"},
             {"role": "assistant", "content": '<tool_call>{"tool":"currency","args":{"amount":200,"from":"EUR","to":"USD"}}</tool_call>'}
         ],
         "expected": '<tool_call>{"tool":"currency","args":{"amount":200,"from":"EUR","to":"GBP"}}</tool_call>'},
        {"prompt": "whats the wether in paris",       "history": [],
         "expected": '<tool_call>{"tool":"weather","args":{"location":"paris","unit":"C"}}</tool_call>'},
    ]

    import os
    os.makedirs("data", exist_ok=True)
    with open(dev_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"📝  Built dev test → {dev_path}")
    return dev_path


# ── Dedup check vs public test set ────────────────────────────────────
def check_dedup(train_file: str, public_test_file: str):
    """Verify no training prompts collide with the public test set (SHA-256)."""
    import hashlib

    def hashes(path):
        s = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    # Handle both formats
                    if "prompt" in obj:
                        prompt = obj["prompt"]
                    elif "messages" in obj:
                        user_msgs = [m["content"] for m in obj["messages"] if m["role"] == "user"]
                        prompt = user_msgs[-1] if user_msgs else ""
                    else:
                        prompt = ""
                    s.add(hashlib.sha256(prompt.encode()).hexdigest())
        return s

    train_h    = hashes(train_file)
    public_h   = hashes(public_test_file)
    collisions = train_h & public_h

    if collisions:
        print(f"⚠️  WARNING: {len(collisions)} training prompt(s) collide with public test set!")
        print("   Remove them from data/train.jsonl before submission.")
    else:
        print("✅  Dedup check passed — zero training prompts overlap with public test set.")


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pocket-Agent")
    parser.add_argument("--test_file",   default=None,
                        help="Path to JSONL test file (default: auto-build dev set)")
    parser.add_argument("--dedup_check", default=None,
                        help="Path to public_test.jsonl to run dedup check")
    args = parser.parse_args()

    if args.dedup_check:
        check_dedup("data/train.jsonl", args.dedup_check)

    test_file = args.test_file or build_dev_test()
    evaluate(test_file)
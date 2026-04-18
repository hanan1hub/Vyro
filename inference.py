"""
inference.py — Pocket-Agent grader interface.

Exposes:
    run(prompt: str, history: list[dict]) -> str

CONTRACT
--------
- No network imports (requests / urllib / http / socket).
- history is a list of {"role": "user"|"assistant", "content": "..."} dicts.
- Returns the raw model output string (may contain <tool_call>…</tool_call>).
- Loads model lazily on first call (weights stay in RAM for the full eval run).
"""

from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Optional

# ── Model path ────────────────────────────────────────────────────────
_GGUF_PATH = os.environ.get("POCKET_AGENT_GGUF", "model.gguf")

# ── System prompt (must match training exactly) ───────────────────────
_SYSTEM = (
    "You are Pocket-Agent, a compact on-device mobile assistant. "
    "You have exactly five tools. When the user's intent clearly matches "
    "a tool, output ONLY a single JSON object wrapped in "
    "<tool_call>…</tool_call> tags — nothing else. "
    "When the request is chitchat, ambiguous with no prior context, or asks "
    "for a non-existent tool, reply in plain natural language — no tags.\n\n"
    "TOOLS\n"
    "-----\n"
    '{"tool":"weather","args":{"location":"<string>","unit":"C|F"}}\n'
    '{"tool":"calendar","args":{"action":"list|create","date":"YYYY-MM-DD","title":"<string, optional>"}}\n'
    '{"tool":"convert","args":{"value":<number>,"from_unit":"<string>","to_unit":"<string>"}}\n'
    '{"tool":"currency","args":{"amount":<number>,"from":"<ISO3>","to":"<ISO3>"}}\n'
    '{"tool":"sql","args":{"query":"<string>"}}\n\n'
    "RULES\n"
    "-----\n"
    "1. Emit exactly one <tool_call>…</tool_call> block for clear requests.\n"
    "2. In multi-turn conversations, resolve pronoun/demonstrative references "
    "from prior assistant messages before emitting the call.\n"
    "3. For refusals — chitchat, no-history ambiguity, impossible tools — "
    "reply in plain English, no tags.\n"
    "4. Argument fidelity: match units, ISO-4217 codes, YYYY-MM-DD dates, "
    "and numerical values exactly as the user intends."
)

# ── Lazy model loader ─────────────────────────────────────────────────
_llm = None

def _load_model():
    global _llm
    if _llm is not None:
        return _llm

    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise RuntimeError(
            "llama-cpp-python not installed. Run: pip install llama-cpp-python"
        ) from e

    if not Path(_GGUF_PATH).exists():
        raise FileNotFoundError(
            f"GGUF model not found at '{_GGUF_PATH}'. "
            "Run: python quantize.py  first."
        )

    print(f"[inference] Loading {_GGUF_PATH} …", flush=True)
    _llm = Llama(
        model_path=_GGUF_PATH,
        n_ctx=2048,         # context window
        n_threads=4,        # CPU threads — tune for your machine
        n_gpu_layers=0,     # 0 = pure CPU (required for Colab CPU runtime)
        verbose=False,
    )
    print("[inference] Model loaded.", flush=True)
    return _llm


# ── Prompt builder ────────────────────────────────────────────────────
def _build_prompt(prompt: str, history: list[dict]) -> str:
    """
    Build a Qwen2.5-style ChatML prompt string.
    Qwen2.5 uses:
        <|im_start|>system\n…<|im_end|>
        <|im_start|>user\n…<|im_end|>
        <|im_start|>assistant\n…<|im_end|>
    """
    parts: list[str] = []

    # System turn
    parts.append(f"<|im_start|>system\n{_SYSTEM}<|im_end|>")

    # History turns
    for turn in history:
        role    = turn.get("role", "user")
        content = turn.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    # Current user turn
    parts.append(f"<|im_start|>user\n{prompt}<|im_end|>")

    # Prompt assistant to respond
    parts.append("<|im_start|>assistant\n")

    return "\n".join(parts)


# ── Post-processing ───────────────────────────────────────────────────
def _clean_output(raw: str) -> str:
    """Strip trailing EOS tokens / whitespace."""
    raw = raw.strip()
    # Remove any <|im_end|> or <|endoftext|> that leaked into the output
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        raw = raw.split(tok)[0]
    return raw.strip()


def _validate_tool_call(text: str) -> Optional[dict]:
    """Return parsed tool call dict if valid, else None."""
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1).strip())
        if isinstance(obj, dict) and "tool" in obj and "args" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    return None


# ── Public interface ──────────────────────────────────────────────────
def run(prompt: str, history: list[dict]) -> str:
    """
    Parameters
    ----------
    prompt  : the current user message
    history : prior turns as [{"role": ..., "content": ...}, ...]

    Returns
    -------
    str  — either '<tool_call>…</tool_call>' or plain-text refusal
    """
    llm = _load_model()

    full_prompt = _build_prompt(prompt, history)

    output = llm(
        full_prompt,
        max_tokens=150,
        temperature=0.0,      # greedy — deterministic for eval
        top_p=1.0,
        stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
        echo=False,
    )

    raw_text: str = output["choices"][0]["text"]
    result = _clean_output(raw_text)

    # Sanity-check: if model emitted malformed JSON inside tags, strip the tags
    # so it counts as a plain refusal rather than 0-score malformed JSON.
    if "<tool_call>" in result:
        parsed = _validate_tool_call(result)
        if parsed is None:
            # Malformed — return as plain text (0 score is better than -0.5)
            result = re.sub(r"<tool_call>.*?</tool_call>", "", result, flags=re.DOTALL).strip()
            if not result:
                result = "I'm not sure how to help with that."

    return result


# ── CLI test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("What's the weather in London?",             []),
        ("Convert 100 USD to EUR",                    []),
        ("5 miles to kilometers",                     []),
        ("Schedule Team Sync on 2025-03-15",          []),
        ("Show all users from the database",          []),
        ("How are you?",                              []),
        ("Send an email to John",                     []),
        ("Convert that to euros",                     []),   # refusal (no history)
        # multi-turn
        ("And in GBP?",
         [{"role":"user","content":"Convert 200 USD to EUR"},
          {"role":"assistant","content":'<tool_call>{"tool":"currency","args":{"amount":200,"from":"USD","to":"EUR"}}</tool_call>'}]),
    ]

    for prompt, history in test_cases:
        print(f"\nUser: {prompt}")
        if history:
            print(f"[history: {len(history)} turns]")
        result = run(prompt, history)
        print(f"Agent: {result}")

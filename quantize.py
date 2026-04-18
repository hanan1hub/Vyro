#!/usr/bin/env python3
"""
quantize.py — Convert merged HuggingFace model → GGUF Q4_K_M.

Steps
-----
1. Clone llama.cpp (if not present)
2. Build llama.cpp convert_hf_to_gguf.py
3. Convert merged/ to FP16 GGUF
4. Quantize to Q4_K_M  →  model.gguf  (≤ 300 MB for 0.5B model)

Run: python quantize.py
"""
import os, sys, subprocess, shutil
from pathlib import Path

MERGED_DIR   = "merged"
LLAMA_DIR    = "llama.cpp"
FP16_GGUF    = "model_fp16.gguf"
FINAL_GGUF   = "model.gguf"
QUANT_TYPE   = "Q4_K_M"   # ~270 MB for 0.5B  — passes both 500 MB gate + 250 MB bonus

def run(cmd, **kwargs):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kwargs)
    return result

# ── 1. Clone llama.cpp ────────────────────────────────────────────────
if not Path(LLAMA_DIR).exists():
    print("📦  Cloning llama.cpp …")
    run(["git", "clone", "--depth", "1",
         "https://github.com/ggerganov/llama.cpp", LLAMA_DIR])
else:
    print("📦  llama.cpp already cloned — skipping")

# ── 2. Install llama.cpp Python deps ─────────────────────────────────
print("📦  Installing llama.cpp convert dependencies …")
req_path = os.path.join(LLAMA_DIR, "requirements", "requirements-convert_hf_to_gguf.txt")
if not os.path.exists(req_path):
    # Older layout
    req_path = os.path.join(LLAMA_DIR, "requirements.txt")
run([sys.executable, "-m", "pip", "install", "-q", "-r", req_path])

# ── 3. Convert merged HF model → FP16 GGUF ───────────────────────────
convert_script = os.path.join(LLAMA_DIR, "convert_hf_to_gguf.py")
if not os.path.exists(convert_script):
    convert_script = os.path.join(LLAMA_DIR, "convert-hf-to-gguf.py")

print(f"🔄  Converting {MERGED_DIR} → {FP16_GGUF} …")
run([
    sys.executable, convert_script,
    MERGED_DIR,
    "--outfile", FP16_GGUF,
    "--outtype", "f16",
])

# ── 4. Build llama-quantize binary ───────────────────────────────────
quantize_bin = os.path.join(LLAMA_DIR, "build", "bin", "llama-quantize")
if not os.path.exists(quantize_bin):
    # Try legacy location
    quantize_bin = os.path.join(LLAMA_DIR, "quantize")

if not os.path.exists(quantize_bin):
    print("🔨  Building llama.cpp quantize binary …")
    build_dir = os.path.join(LLAMA_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)
    run(["cmake", "..", "-DLLAMA_NATIVE=OFF"], cwd=build_dir)
    run(["cmake", "--build", ".", "--config", "Release", "--target", "llama-quantize", "-j4"],
        cwd=build_dir)
    quantize_bin = os.path.join(build_dir, "bin", "llama-quantize")

# ── 5. Quantize FP16 GGUF → Q4_K_M ──────────────────────────────────
print(f"⚡  Quantizing to {QUANT_TYPE} → {FINAL_GGUF} …")
run([quantize_bin, FP16_GGUF, FINAL_GGUF, QUANT_TYPE])

# ── 6. Report size ────────────────────────────────────────────────────
size_mb = Path(FINAL_GGUF).stat().st_size / (1024 * 1024)
print(f"\n✅  Quantization complete")
print(f"    {FINAL_GGUF}: {size_mb:.1f} MB")

if size_mb <= 250:
    print("    🏆  Qualifies for ≤ 250 MB bonus!")
elif size_mb <= 500:
    print("    ✅  Passes hard gate (≤ 500 MB)")
else:
    print("    ⚠️  WARNING: Exceeds 500 MB hard gate! "
          "Try a smaller quant: Q3_K_M or Q2_K")
    print("    Rerun with: QUANT_TYPE='Q2_K' and update this script")

# Clean up FP16 to save disk
if Path(FP16_GGUF).exists():
    os.remove(FP16_GGUF)
    print(f"    🗑️  Removed intermediate {FP16_GGUF}")

print("\nNext step: python chatbot.py  or  python eval.py")

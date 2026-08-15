#!/usr/bin/env python3
"""
Validates the Phase 3 streamer output against reference_output.txt.

The streamer uses llama_token_to_piece (lstrip=0), while the reference was
extracted from llama-cli with a different detokenization pipeline. Therefore,
the comparison normalizes whitespace (ignores redundant spaces/line breaks) to
compare token-by-token CONTENT, not the exact rendering.

usage: python validate_phase3.py [output_file]
"""

import re
import sys

REFERENCE = "../doc/reference_output.txt"
DEFAULT_OUTPUT = "../outputs/phase3/phase3_output.txt"


def normalize(text: str) -> str:
    """Collapse any run of whitespace into 1 space (keep separation)."""
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT

    with open(REFERENCE) as f:
        ref = f.read()
    with open(out_path) as f:
        out = f.read()

    ref_n = normalize(ref)
    out_n = normalize(out)

    print("=" * 60)
    print("REFERENCE (normalized):")
    print(ref_n[:300] + ("..." if len(ref_n) > 300 else ""))
    print("=" * 60)
    print("STREAMER (normalized):")
    print(out_n[:300] + ("..." if len(out_n) > 300 else ""))
    print("=" * 60)

    if ref_n == out_n:
        print("✅ PHASE 3 PASS: output IDENTICAL to reference (token-by-token).")
        return

    # diagnostica a primeira divergência
    L = min(len(ref_n), len(out_n))
    div = next((i for i in range(L) if ref_n[i] != out_n[i]), L)
    print(f"❌ Divergence at char {div} (len ref={len(ref_n)}, len out={len(out_n)})")
    print(f"   ref ctx: {repr(ref_n[max(0, div-40):div+40])}")
    print(f"   out ctx: {repr(out_n[max(0, div-40):div+40])}")
    print("\nNote: if the divergence is only spacing/whitespace, it's a rendering")
    print("issue (not a streaming error). If it's content, investigate the rebind.")


if __name__ == "__main__":
    main()
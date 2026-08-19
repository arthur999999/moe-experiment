#!/usr/bin/env python3
"""check_dense_residency.py -- gate 3: with dense=anon + dontneed=1, the dense
(non-expert) file pages must NOT stay in the page cache after the run.

Protocol (sweep -> anon run -> check; compare vs a mmap run):
  python scripts/check_dense_residency.py MODEL sweep
  ./build/phase6_5_stream MODEL PROMPT 1024 4 42 1024 4 1 1 0 2 0 32 >/dev/null
  python scripts/check_dense_residency.py MODEL check
  # same check after a dense=mmap run: dense resident should be HIGH (control)
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_cache_purity as ccp   # parse_gguf / sweep_file / group_stats reused

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model"); ap.add_argument("mode", choices=["sweep","check"])
    ap.add_argument("--dense-threshold", type=float, default=15.0,
                    help="dense resident%% below this -> PURE (default 15)")
    ap.add_argument("--expert-match", default="exps")
    args = ap.parse_args()
    if args.mode == "sweep":
        ccp.sweep_file(args.model); return 0

    ranges = ccp.parse_gguf(args.model)
    dense  = {n: r for n, r in ranges.items() if args.expert_match not in n}
    expert = {n: r for n, r in ranges.items() if args.expert_match in n}
    if not dense:
        print("error: no dense tensors (every tensor matched 'exps'?)"); return 2

    nd, td, rd = ccp.group_stats(dense, args.model)
    ne, te, re = ccp.group_stats(expert, args.model)
    pct = lambda a, b: 100.0 * a / b if b else 0.0
    print("== check_dense_residency ==")
    print(f"model: {args.model}")
    print(f"{'group':<10}{'tensors':>8}{'total MB':>12}{'resident MB':>14}{'resident%':>11}")
    print("-" * 55)
    print(f"{'dense':<10}{nd:>8}{td/1e6:>12.1f}{rd/1e6:>14.1f}{pct(rd,td):>10.1f}%")
    print(f"{'expert':<10}{ne:>8}{te/1e6:>12.1f}{re/1e6:>14.1f}{pct(re,te):>10.1f}%")
    df = pct(rd, td)
    if df < args.dense_threshold:
        print(f"\nverdict: PURE  (dense resident {df:.1f}% < {args.dense_threshold}%)")
        print("         -> anon DONTNEED stuck: nothing refaults the dense mmap.")
        return 0
    print(f"\nverdict: NOT PURE (dense resident {df:.1f}% >= {args.dense_threshold}%)")
    print("         causes: ran dense=mmap/warm, dontneed=0, missed a dense tensor,")
    print("         or a stray path still reads the dense mmap (check majflt in the run).")
    return 1

if __name__ == "__main__":
    sys.exit(main())
"""Filter blind-generator output into pools and handle the hand audit.

    uv run --python 3.12 --with sentence-transformers python scripts/harder_filter.py filter
    python scripts/harder_filter.py apply-audit

`filter` writes data_gen/pools/*.jsonl, data_gen/rejected.jsonl and data_gen/audit_sheet.md.
Tick bad items on the sheet, then `apply-audit` removes them. `scene-fix` applies the
round-2 keep/fix/drop answers (data_gen/raw/fix-*.txt) to the harmless pool, re-filters it
and writes audit_sheet_2.md; `apply-audit-2` applies that sheet. See data_gen/CONTEXT.md.
"""

import argparse
import json

from toolguard.harder.audit import apply_audit, refilter_benign, run_filter
from toolguard.harder.scene_fix import apply_scene_fix

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def make_embedder():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    return lambda texts: model.encode(texts, batch_size=64, show_progress_bar=False).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("filter", "apply-audit", "scene-fix", "apply-audit-2"))
    p.add_argument("--root", default="data_gen")
    p.add_argument("--no-embed", action="store_true", help="skip the embedding-similarity check")
    args = p.parse_args()
    if args.command == "filter":
        counts = run_filter(args.root, None if args.no_embed else make_embedder())
        print(json.dumps(counts, indent=2))
    elif args.command == "scene-fix":
        decisions = apply_scene_fix(args.root)
        counts = refilter_benign(args.root, None if args.no_embed else make_embedder(), "audit_sheet_2.md")
        print(json.dumps({"decisions": decisions, **counts}, indent=2))
    elif args.command == "apply-audit-2":
        removed = apply_audit(args.root, "audit_sheet_2.md", "audit_2.md")
        print(f"removed {len(removed)} audited items")
    else:
        removed = apply_audit(args.root)
        print(f"removed {len(removed)} audited items")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the adversarial multi-agent critique of the probe-design algorithm.

Examples
--------
    python run_critique.py                 # text report to stdout
    python run_critique.py --format markdown --out critique_report.md
"""
from __future__ import annotations

import argparse
import sys

from primer2.critique import CritiqueContext, default_panel


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Multi-agent algorithm critique")
    parser.add_argument("--format", choices=["text", "markdown"], default="text")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="write the report to this file as well")
    args = parser.parse_args(argv)

    context = CritiqueContext.default(seed=args.seed)
    panel = default_panel()
    report = panel.run(context)

    rendered = report.to_markdown() if args.format == "markdown" else report.to_text()
    print(rendered)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(report.to_markdown())
        print("wrote %s" % args.out, file=sys.stderr)

    counts = report.counts_by_severity()
    from primer2.critique import Severity
    # Non-zero exit if any CRITICAL finding, so this can gate CI if desired.
    return 1 if counts.get(Severity.CRITICAL, 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Figure: a produced reasoning walk and its rendered report, showing ONLY the report-determining steps so that
every field in the report traces to a shown answer (organ, procedure, histologic type, and the grade-component
scores). Routine intermediate steps are elided with an ellipsis.

  python gow/eval/make_walker_fig.py
"""
import json, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, GRAY_F, GRAY_E, BLUE_F, BLUE_E, ACC = "#1F2933", "#F3F4F6", "#9AA5B1", "#E7EEF6", "#3B6EA5", "#1a7a4a"
KEYS = ["what is the organ", "what is the procedure", "histologic type",
        "tubular", "nuclear pleomorphism", "mitotic"]


def pick():
    for p in json.load(open("gow/artifacts/test1_predictions.json")):
        ch = p["chain-of-thought"]
        org = next((s["answer"] for s in ch if s["question"].lower().startswith("what is the organ")), "")
        rep = next((s["answer"] for s in ch if "final pathology report" in s["question"].lower()), "")
        has_grade = any("mitotic" in s["question"].lower() or "nuclear pleomorphism" in s["question"].lower() for s in ch)
        if org == "Breast" and rep and has_grade:
            return ch, rep
    raise SystemExit("no suitable breast case found")


def main():
    chain, report = pick()
    steps, seen = [], set()
    for s in chain:
        ql = s["question"].strip().lower()
        if any(k in ql for k in KEYS) and ql not in seen:
            seen.add(ql); steps.append((s["question"].strip(), s["answer"].strip()))

    def wrap(t, n): return "\n".join(textwrap.wrap(t, n))
    rep_clean = report.replace("\\n", " ").replace("  ", " ").strip()

    n = len(steps)
    fig, ax = plt.subplots(figsize=(5.0, 0.92 * (n + 2.4)))
    ax.set_xlim(0, 10); ax.set_ylim(0, n + 2.4); ax.axis("off")
    y = n + 2.0
    dy = 1.0
    for i, (q, a) in enumerate(steps):
        yb = y - i * dy
        ax.add_patch(FancyBboxPatch((0.3, yb - 0.78), 9.4, 0.74, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    fc=GRAY_F, ec=GRAY_E, lw=1.0))
        ax.text(0.55, yb - 0.30, "Q  " + wrap(q, 52).split("\n")[0], fontsize=7.6, va="center", color=INK)
        ax.text(0.55, yb - 0.60, "A  " + wrap(a, 52).split("\n")[0], fontsize=7.6, va="center", color=ACC, weight="bold")
        ax.add_patch(FancyArrowPatch((5, yb - 0.80), (5, yb - 1.02), arrowstyle="-|>", mutation_scale=9, color="#666", lw=1.0))
        if i == 1:  # ellipsis after procedure: routine intermediate steps elided
            ax.text(5, yb - 0.92, ".  .  .", ha="center", va="center", fontsize=9, color="#999")
    # report box
    yb = y - n * dy - 0.15
    ax.add_patch(FancyBboxPatch((0.3, yb - 1.05), 9.4, 1.0, boxstyle="round,pad=0.03,rounding_size=0.08",
                                fc=BLUE_F, ec=BLUE_E, lw=1.3))
    ax.text(0.55, yb - 0.30, "Rendered report (deterministic)", fontsize=7.6, va="center", color=BLUE_E, weight="bold")
    for j, line in enumerate(textwrap.wrap(rep_clean[:170], 56)[:3]):
        ax.text(0.55, yb - 0.55 - j * 0.24, line, fontsize=6.9, va="center", color=INK)
    fig.tight_layout(pad=0.2); fig.savefig("paper/figures/walker_example.pdf")
    print(f"[saved] paper/figures/walker_example.pdf  ({n} report-determining steps)")
    print("report:", rep_clean[:130])


if __name__ == "__main__":
    main()

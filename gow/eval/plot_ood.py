#!/usr/bin/env python3
"""
Figure: generalized distribution-shift detector operating point. As the per-organ Mahalanobis threshold is
scaled, in-distribution false-positive rate (held-out challenge slides) falls while novel-organ recall (public
TCGA organs outside the seven) stays high. The shipped operating point (scale 2.5) flags ~2% of in-distribution
slides yet still catches the dominant uterus shift and 9 of 10 further novel organs. Numbers are the measured
sweep (gow/eval/sweep_ood_threshold.py) over 1119 held-out in-dist slides + 10 public TCGA novel organs.

  python gow/eval/plot_ood.py
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# measured sweep: (threshold scale, in-dist FP %, novel-organ recall out of 10)
SCALE = [1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50, 4.00]
FP    = [13.8, 8.5,  6.0,  4.6,  3.4,  2.0,  1.3,  1.1,  0.8]
REC   = [10,   10,   9,    9,    9,    9,    9,    9,    8]
SHIP  = 2.50
BLUE, ACCENT, GREY = "#3B6EA5", "#C1553B", "#555555"


def main():
    out = "paper/figures/ood_separation.pdf"
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})
    fig, axL = plt.subplots(figsize=(4.8, 2.8))
    axR = axL.twinx()
    axL.plot(SCALE, FP, "-o", color=ACCENT, lw=1.7, ms=4)
    axR.plot(SCALE, [r * 10 for r in REC], "-s", color=BLUE, lw=1.7, ms=4)
    axL.axvline(SHIP, color=GREY, ls="--", lw=1.2)
    axL.annotate("shipped\n(2\\% FP, 9/10)", xy=(SHIP, 2.0), xytext=(2.72, 8.2),
                 fontsize=7.5, color=GREY, arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9))
    axL.set_xlabel("per-organ threshold scale")
    axL.set_ylabel("in-distribution FP (\\%)", color=ACCENT)
    axR.set_ylabel("novel-organ recall (\\%)", color=BLUE)
    axL.tick_params(axis="y", colors=ACCENT); axR.tick_params(axis="y", colors=BLUE)
    axL.set_ylim(0, 15); axR.set_ylim(60, 102); axL.set_xlim(0.9, 4.1)
    for ax in (axL, axR):
        ax.spines["top"].set_visible(False)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); print(f"[saved] {out}")


if __name__ == "__main__":
    main()

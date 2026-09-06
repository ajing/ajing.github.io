"""Build the article's SVG figures. Requires Matplotlib, no model calls.

Usage: python3 plot_results.py --output-dir ./figures
Input: results.json beside this script.
"""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch


def protocol(out):
    fig, ax = plt.subplots(figsize=(7.6, 9.6))
    fig.patch.set_facecolor("#fafaf7")
    ax.set(xlim=(0, 10), ylim=(0, 12))
    ax.axis("off")
    ink, gray = "#16343b", "#52656b"
    ax.text(0.4, 11.75, "What changed in the visibility study", fontsize=18,
            weight="bold", color=ink)
    ax.text(0.4, 11.25, "Same starting evidence. New actions in every branch.",
            fontsize=13, color=gray)

    def box(y, title, lines, fill, height=1.45):
        ax.add_patch(FancyBboxPatch((0.4, y), 9.2, height, boxstyle="round,pad=0.03,rounding_size=0.12",
                                   facecolor=fill, edgecolor="#cbd5d5", linewidth=1))
        ax.text(0.7, y + height - 0.4, title, fontsize=15, weight="bold", color=ink)
        ax.text(0.7, y + height - 0.78, lines, fontsize=12.5, color=gray, va="top", linespacing=1.45)

    def arrow(top, bottom):
        ax.annotate("", (5, bottom), (5, top),
                    arrowprops={"arrowstyle": "-|>", "color": "#607d83", "lw": 1.8})

    box(9.35, "Freeze one common prefix per question", "Neutral plan + four initial searches\n+ four workers' private reports", "#e6eeee")
    arrow(9.3, 8.95)
    box(7.65, "P  |  Private exploration", "Own passages + own report + common plan", "#eef2f1", 1.25)
    box(6.2, "E  |  Shared evidence", "P's view + all initial raw passages and queries", "#e1eeec", 1.25)
    box(4.75, "H  |  Shared reports", "E's raw evidence + peers' candidate reports", "#f0eade", 1.25)
    ax.text(5, 4.42, "Alternative branches; same worker rule", ha="center", color=gray, fontsize=12)
    arrow(4.2, 3.85)
    box(2.45, "Choose and execute four new actions", "The visibility intervention changes actual search.\nNo replay of another branch's next queries.", "#e6eeee")
    arrow(2.4, 2.05)
    box(0.65, "Synthesize from each branch's raw evidence", "Same final prompt; no peer reports or arm label.\nP also pools all team evidence at this stage.", "#e1eeec")
    ax.text(0.4, 0.17, "S is a separate adaptive sequential-agent comparison.", fontsize=11.5, color=gray)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    fig.savefig(out / "visibility-protocol.svg", facecolor=fig.get_facecolor(), metadata={"Date": None})
    plt.close(fig)


def outcomes(data, out):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.patch.set_facecolor("#fafaf7")
    ax.set_facecolor("#fafaf7")
    colors = {"correct": "#17776c", "wrong": "#b54735", "unknown": "#dfe5e4"}
    for y, arm in [(1, "B"), (0, "V")]:
        row = data["last_action"]["aggregates"][arm]
        left = 0
        for outcome in ("correct", "wrong", "unknown"):
            count = row[outcome]
            if count:
                ax.barh(y, count, left=left, height=.45, color=colors[outcome], edgecolor="white", linewidth=1.5)
                ax.text(left + count / 2, y, str(count), ha="center", va="center", fontsize=15,
                        weight="bold", color="#16343b" if outcome == "unknown" else "white")
            left += count
        assert left == 16
    ax.set_yticks([1, 0], ["B · Free choice", "V · Verification priority"], fontsize=12)
    ax.set_xlim(0, 16)
    ax.set_ylim(-.6, 1.6)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_xlabel("Trajectory count (16 per policy)", fontsize=11, color="#52656b", labelpad=9)
    ax.tick_params(length=0, labelcolor="#16343b", pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.05, .93, "A small gain, with an additional error", fontsize=18, weight="bold", color="#16343b")
    fig.text(.05, .855, "Eight reused development questions, two repetitions per policy", fontsize=12, color="#52656b")
    fig.legend(handles=[Patch(color=colors[k], label=label) for k, label in
                        [("correct", "Correct"), ("wrong", "Wrong submission"), ("unknown", "UNKNOWN")]],
               loc="lower left", bbox_to_anchor=(.05, .02), ncol=3, frameon=False, fontsize=11)
    fig.text(.05, .005, "Descriptive counts; repetitions are not independent tasks.", fontsize=10, color="#52656b")
    fig.subplots_adjust(left=.31, right=.96, bottom=.27, top=.78)
    fig.savefig(out / "last-action-outcomes.svg", facecolor=fig.get_facecolor(), metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "swarm-blog-2026"})
    data = json.loads(Path(__file__).with_name("results.json").read_text())
    protocol(args.output_dir)
    outcomes(data, args.output_dir)
    for name in ("visibility-protocol.svg", "last-action-outcomes.svg"):
        path = args.output_dir / name
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print("Wrote visibility-protocol.svg and last-action-outcomes.svg")

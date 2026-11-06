"""Figures for the write-up, drawn from the analysis outputs.

  validation   validation reward at each evaluation step (training judge)
  headline     base vs RL mean reward on the test prompts, per judge and split, with 95%
               prompt-bootstrap intervals
  parts        share of answers earning each part of the reward, base vs RL, with 95%
               cluster-bootstrap intervals; code checks over all answers, judge parts over
               answers that rendered
  examples     12 random test prompts, base and RL side by side (needs sample.py and render.py)

Each figure answers one question and its title states the answer; method details
(judges, sample sizes, intervals) belong in the caption that accompanies it. All four
share one layout: the same width, title position, legend and margins.

Reads OUT_DIR/headline.json, OUT_DIR/per_item.json and DATA_DIR/selection.json, so run
results.py and per_item.py first. Writes PNG files to OUT_DIR/figures/.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import DATA_DIR, OUT_DIR  # noqa: E402

# Two data colours only: solid grey for the base model, black for the RL model.
INK, SECONDARY, RULE, SURFACE = "#111111", "#4D4D4D", "#D9D9D9", "#FFFFFF"
BASE, RL = "#A6A6A6", "#111111"

WIDTH, DPI = 6.4, 300  # inches; matches a blog text column, sharp on high-density screens
LEFT_PAD = 0.03  # title and legend start here, in figure fraction
HEADER_IN = 0.78  # inches above the plot for title and legend

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "text.color": INK, "axes.labelcolor": SECONDARY, "axes.labelsize": 8.5,
    "xtick.color": SECONDARY, "ytick.color": INK, "xtick.labelsize": 8.5, "ytick.labelsize": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def new_figure(plot_height_in, left, right=0.94, header_in=HEADER_IN, bottom_in=0.12):
    """A figure whose plot is plot_height_in tall, with the shared header band above it."""
    height = header_in + plot_height_in + bottom_in
    fig, ax = plt.subplots(figsize=(WIDTH, height))
    fig.subplots_adjust(left=left, right=right, top=1 - header_in / height, bottom=bottom_in / height)
    return fig, ax


def header(fig, text, with_legend=True):
    h = fig.get_size_inches()[1]
    fig.text(LEFT_PAD, 1 - 0.14 / h, text, fontsize=11.5, fontweight="semibold", color=INK, ha="left", va="top")
    if with_legend:
        handles = [Line2D([], [], linestyle="none", marker="s", markersize=7, markerfacecolor=c, markeredgecolor=c,
                          label=l) for l, c in (("Base model", BASE), ("After RL", RL))]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(LEFT_PAD - 0.008, 1 - 0.52 / h), ncol=2,
                   frameon=False, fontsize=8.5, handletextpad=0.35, columnspacing=1.2, labelcolor=SECONDARY)


def bare(ax):
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)


def save(fig, name):
    out = OUT_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=DPI)
    plt.close(fig)
    print(f"wrote figures/{name}.png")


def whisker(ax, y, ci, height):
    """A thin 95% interval line with short end caps, centred on a bar."""
    lo, hi = ci
    ax.plot([lo, hi], [y, y], color=INK, linewidth=0.7, solid_capstyle="butt", zorder=3)
    ax.plot([lo, lo, None, hi, hi], [y - height / 2, y + height / 2, None, y - height / 2, y + height / 2],
            color=INK, linewidth=0.7, zorder=3)


def paired_bars(ax, rows, fmt, xmax, change=None):
    """rows: dicts with label, base, rl, optional base_ci and rl_ci (95% intervals, drawn as
    whiskers), and optional bold or heading. change(base, rl) -> text adds a "Change" column
    to the right of the plot."""
    bar, gap, row_gap = 0.26, 0.05, 0.36
    y, ticks, labels, bold = 0.0, [], [], []
    for row in rows:
        if row.get("heading"):
            y -= 0.10
            ax.annotate(row["label"], (LEFT_PAD, y), xycoords=("figure fraction", "data"), ha="left", va="center",
                        fontsize=9, fontweight="semibold", color=INK, annotation_clip=False)
            y -= 0.50
            continue
        top, bottom = y - bar / 2, y - bar - gap - bar / 2
        ax.barh(top, row["base"], height=bar, color=BASE, linewidth=0)
        ax.barh(bottom, row["rl"], height=bar, color=RL, linewidth=0)
        ends = {}
        for key, yy in (("base", top), ("rl", bottom)):
            ci = row.get(f"{key}_ci")
            if ci:
                whisker(ax, yy, ci, bar * 0.45)
            ends[key] = max(row[key], ci[1] if ci else 0)
        ax.text(ends["base"] + xmax * 0.012, top, fmt(row["base"]), va="center", ha="left", fontsize=8,
                color=SECONDARY)
        ax.text(ends["rl"] + xmax * 0.012, bottom, fmt(row["rl"]), va="center", ha="left", fontsize=8.5,
                color=INK, fontweight="semibold")
        if change:
            ax.annotate(change(row["base"], row["rl"]), (1.0, (top + bottom) / 2), xycoords=("axes fraction", "data"),
                        xytext=(58, 0), textcoords="offset points", ha="right", va="center", fontsize=8.5,
                        color=INK, annotation_clip=False)
        ticks.append((top + bottom) / 2)
        labels.append(row["label"])
        bold.append(row.get("bold", False))
        y = bottom - bar / 2 - row_gap
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    for tick, b in zip(ax.get_yticklabels(), bold):
        if b:
            tick.set_fontweight("semibold")
    if change:
        ax.annotate("Change", (1.0, 0.05), xycoords=("axes fraction", "data"), xytext=(58, 0),
                    textcoords="offset points", ha="right", va="bottom", fontsize=8, color=SECONDARY,
                    annotation_clip=False)
    ax.set_ylim(y + row_gap - 0.05, 0.12)
    ax.set_xlim(0, xmax)
    ax.set_xticks([])
    ax.tick_params(axis="y", pad=8)
    bare(ax)


def validation():
    scores = json.loads((DATA_DIR / "selection.json").read_text(encoding="utf-8"))["scores"]
    steps = sorted(int(s) for s in scores)
    values = [scores[str(s)] for s in steps]
    fig, ax = new_figure(2.3, left=0.115, header_in=0.50, bottom_in=0.42)
    header(fig, "Validation reward rose during RL", with_legend=False)
    for v in (0.25, 0.5, 0.75, 1.0):
        ax.axhline(v, color=RULE, linewidth=0.6, zorder=0)
    ax.axhline(0, color=SECONDARY, linewidth=0.8, zorder=0)
    ax.plot(steps, values, color=RL, linewidth=1.6, solid_capstyle="round", solid_joinstyle="round", zorder=2)
    ax.scatter(steps[1:], values[1:], s=14, color=RL, zorder=3)
    ax.scatter([steps[0]], [values[0]], s=26, color=BASE, edgecolor=SURFACE, linewidth=1.2, zorder=4)
    ax.annotate(f"Base model {values[0]:.2f}", (steps[0], values[0]), xytext=(7, -1), textcoords="offset points",
                ha="left", va="top", fontsize=8, color=SECONDARY)
    ax.annotate(f"{values[-1]:.2f}", (steps[-1], values[-1]), xytext=(0, 7), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="semibold")
    ax.set_xlim(-4, steps[-1] + 5)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(0, steps[-1] + 1, 30))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"], color=SECONDARY, fontsize=8)
    ax.set_xlabel("Training step", labelpad=5)
    ax.set_ylabel("Validation reward", labelpad=6)
    bare(ax)
    ax.tick_params(axis="x", pad=4)
    save(fig, "validation")


def headline():
    data = json.loads((OUT_DIR / "headline.json").read_text(encoding="utf-8"))
    split_name = {"test": "Test prompts", "test_unseen": "Held-out category (space)"}

    def bars(section):
        return [{"label": split_name[r["split"]], "base": r["base"], "rl": r["trained"],
                 "base_ci": r["base_ci95"], "rl_ci": r["trained_ci95"]} for r in data[section]]

    rows = [{"label": "Evaluation judge (Gemini 3.1 Pro)", "heading": True}, *bars("evaluation_judge"),
            {"label": "Training judge (Gemma 3 27B)", "heading": True}, *bars("training_judge")]
    fig, ax = new_figure(1.95, left=0.30, right=0.84)
    header(fig, "RL raised the reward on new prompts")
    paired_bars(ax, rows, lambda v: f"{v:.2f}", xmax=1.08, change=lambda b, r: f"{r - b:+.2f}")
    save(fig, "headline")


def parts():
    data = [e for e in json.loads((OUT_DIR / "per_item.json").read_text(encoding="utf-8"))["parts"] if e["split"] == "test"]
    names = {"valid SVG": "Valid SVG", "not blank": "Renders an image", "moves": "Animates",
             "scene": "Scene recognisable", "objects": "Named objects present", "isometric": "Isometric projection",
             "composition": "Coherent composition", "no figure": "No people or animals",
             "motion": "Motion matches the prompt"}
    rows = []
    for kind, heading in (("code", "Code checks (all answers)"), ("judge", "Judge (answers that rendered)")):
        rows.append({"label": heading, "heading": True})
        rows += [{"label": names[e["part"]], "base": e["base"]["share"], "rl": e["trained"]["share"],
                  "base_ci": e["base"]["ci95"], "rl_ci": e["trained"]["ci95"], "bold": e["part"] == "isometric"}
                 for e in data if e["kind"] == kind]
    fig, ax = new_figure(3.75, left=0.30, right=0.84)
    header(fig, "RL improved every part; isometric drawing stayed rare")
    paired_bars(ax, rows, lambda v: f"{v:.0%}", xmax=1.08,
                change=lambda b, r: f"{round(100 * r) - round(100 * b):+d} pts")
    save(fig, "parts")


def examples():
    """12 random test prompts (sample.py, seed 0), base and RL side by side, frames from render.py.

    Each pair sits in its own card (a hairline frame), with its two pictures close together and a
    wide gutter between cards, so the pairs read as units. The prompt is printed under the pictures,
    wrapped to at most two lines."""
    import textwrap

    from matplotlib.patches import FancyBboxPatch

    sample = OUT_DIR / "random_sample"
    items = json.loads((sample / "index.json").read_text(encoding="utf-8"))["items"]
    cols, tile, inner, pad, label_h, caption_h = 2, 1.40, 0.04, 0.10, 0.22, 0.32
    gutter_x, gutter_y = 0.34, 0.22
    card_w, card_h = 2 * tile + inner + 2 * pad, label_h + tile + caption_h + 2 * pad
    rows = -(-len(items) // cols)
    margin, header_in = 0.20, 0.58
    width = 2 * margin + cols * card_w + (cols - 1) * gutter_x
    height = header_in + rows * card_h + (rows - 1) * gutter_y + 0.14
    fig = plt.figure(figsize=(width, height))
    fig.text(margin / width, 1 - 0.14 / height, "Random test prompts, before and after RL", fontsize=11.5,
             fontweight="semibold", color=INK, ha="left", va="top")
    for k, item in enumerate(items):
        r, c = divmod(k, cols)
        cx = margin + c * (card_w + gutter_x)
        cy_top = height - header_in - r * (card_h + gutter_y)
        fig.patches.append(FancyBboxPatch((cx / width, (cy_top - card_h) / height), card_w / width, card_h / height,
                                          boxstyle="round,pad=0,rounding_size=0.006", transform=fig.transFigure,
                                          facecolor="none", edgecolor=RULE, linewidth=0.8, zorder=0, figure=fig))
        lines = textwrap.wrap(item["prompt"], width=62)
        if len(lines) > 2:
            lines = [lines[0], textwrap.shorten(" ".join(lines[1:]), width=60, placeholder=" ...")]
        fig.text((cx + pad) / width, (cy_top - pad - label_h - tile - 0.07) / height, "\n".join(lines),
                 fontsize=7, color=SECONDARY, ha="left", va="top", linespacing=1.35)
        for j, model in enumerate(("base", "trained")):
            x = cx + pad + j * (tile + inner)
            y_label = cy_top - pad - label_h * 0.42
            fig.text(x / width, y_label / height, "Base model" if model == "base" else "After RL", fontsize=8,
                     color=SECONDARY if model == "base" else INK,
                     fontweight="normal" if model == "base" else "semibold", ha="left", va="center")
            ax = fig.add_axes([x / width, (cy_top - pad - label_h - tile) / height, tile / width, tile / height])
            png = sample / "tiles" / (item[model] or "").replace(".svg", ".png")
            if item[model] and png.exists():
                ax.imshow(plt.imread(png), interpolation="lanczos")
            else:
                ax.text(0.5, 0.5, "Not shown" if not item[model] else "Broken SVG", ha="center", va="center",
                        fontsize=8, color=SECONDARY, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color(RULE)
                side.set_linewidth(0.5)
    save(fig, "examples")


def main() -> int:
    validation()
    headline()
    parts()
    if (OUT_DIR / "random_sample" / "tiles").exists():
        examples()
    else:
        print("skipped figures/examples.png: no random_sample/tiles/ (run sample.py and render.py first)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

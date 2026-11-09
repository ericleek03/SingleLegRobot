"""Graphs from your TensorBoard logs for the single leg on the radial boom.

Reads the event files rsl_rl writes and makes 4 PNG pages + a summary table:
    01_training.png       is training healthy?    reward, episode length, collapses, noise, losses
    02_task.png           is it doing the job?    speed vs command, distance / laps, height, gait, clearance
    03_motor_limits.png   can the motor take it?  RMS torque vs 5.46 Nm, impacts vs 3 BW, power, cost of transport
    04_rewards.png        what is it paid for?    every reward term in its own small panel
    summary.txt           last-20-iteration averages vs targets (also printed)

    python scripts/tb_plots.py                                  # latest run
    python scripts/tb_plots.py --runs 2026-11-06_17-10-02       # a specific run (folder name or full path)
    python scripts/tb_plots.py --runs stage1_dir stage2_dir --labels "stage 1" "stage 2"   # overlay up to 3 runs
    python scripts/tb_plots.py --smooth 0.95                    # smoother lines (0 = raw)

Plots go to logs/plots/<run name>/.
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_ROOT = "logs/rsl_rl/single_leg_direct"
OUT_ROOT = "logs/plots"

# ---- look: one hue per run (fixed order), quiet grey for grids and limit lines ----
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9,
    "xtick.color": INK_3, "ytick.color": INK_3, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9, "font.size": 9,
})

# ---- targets for the summary (edit to taste). (direction, value): "<" = should stay below, ">" = above ----
TARGETS = {
    "torque_rms": ("<", 5.46),      # GIM8108-6 continuous torque [Nm]
    "impact": ("<", 3.0),           # touchdown peak [dog body weights]
    "height_err": ("<", 2.0),       # [cm] off standing height
    "contact": (">", 0.85),         # share of steps where the foot matches the gait clock
    "clear_err": ("<", 2.0),        # [cm] short of the commanded swing height
    "vel_err": ("<", 0.08),         # [m/s] off the commanded forward speed
    "ep_len_s": (">", 9.0),         # [s] of a 10 s episode (lower = it collapses)
}

TAGS = {
    "reward": "Train/mean_reward",
    "ep_len": "Train/mean_episode_length",
    "noise": "Policy/mean_noise_std",
    "value_loss": "Loss/value_function",
    "surrogate": "Loss/surrogate",
    "lr": "Loss/learning_rate",
    "collapsed": "Episode_Termination/collapsed",
    "height_err": "Episode/body_height_error_cm",
    "contact": "Episode/contact_match",
    "clear_err": "Episode/clearance_error_cm",
    "torque_rms": "Episode/torque_rms_max_Nm",
    "impact": "Episode/impact_peak_bw",
    "power": "Episode/mean_power_W",
    "vel_err": "Episode/vel_error_ms",
    "speed": "Episode/mean_speed_ms",
    "cot": "Episode/cost_of_transport",
}


# ------------------------------------------------------------------------------------------------- reading
def find_run(name):
    if name is None:
        runs = [d for d in glob.glob(os.path.join(LOG_ROOT, "*")) if glob.glob(os.path.join(d, "events.out.*"))]
        if not runs:
            raise SystemExit(f"no runs with TensorBoard files in {LOG_ROOT}")
        return max(runs, key=os.path.getmtime)
    if os.path.isdir(name):
        return name
    hits = glob.glob(os.path.join(LOG_ROOT, f"*{name}*"))
    if not hits:
        raise SystemExit(f"run '{name}' not found in {LOG_ROOT}")
    return max(hits, key=os.path.getmtime)


def read_scalars(run_dir):
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})     # 0 = keep every point
    acc.Reload()
    data = {}
    for tag in acc.Tags()["scalars"]:
        ev = acc.Scalars(tag)
        data[tag] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return data


class _Loader(yaml.SafeLoader):
    """Isaac Lab writes tuples as !!python/tuple; read them as lists and ignore other python tags."""


_Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda l, n: l.construct_sequence(n))
_Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda l, s, n: None)


def read_env_cfg(run_dir):
    path = os.path.join(run_dir, "params", "env.yaml")
    try:
        with open(path) as f:
            return yaml.load(f, Loader=_Loader) or {}
    except Exception:
        return {}


def smooth(y, a):
    """TensorBoard-style exponential smoothing."""
    if a <= 0 or len(y) < 2:
        return y
    out, last = np.empty_like(y, dtype=float), y[0]
    for i, v in enumerate(y):
        last = a * last + (1 - a) * v
        out[i] = last
    return out


# ------------------------------------------------------------------------------------------------- drawing
def draw(ax, runs, key, title, ylabel, a, transform=None, logy=False):
    """One metric, every run in its own colour: faint raw line + smoothed line."""
    found = False
    for run in runs:
        tag = TAGS.get(key, key)
        if tag not in run["data"]:
            continue
        x, y = run["data"][tag]
        if transform is not None:
            y = transform(y, run)
        ax.plot(x, y, color=run["color"], lw=0.8, alpha=0.22)
        ax.plot(x, smooth(y, a), color=run["color"], lw=2.0, label=run["label"])
        found = True
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if logy and found:
        ax.set_yscale("log")
    if not found:
        ax.text(0.5, 0.5, "not logged in this run", transform=ax.transAxes, ha="center", va="center", color=INK_3)
    return found


def ref_line(ax, y, text, style="--"):
    ax.axhline(y, color=INK_3, lw=1.2, ls=style, zorder=1)
    ax.annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK_2)


def finish(fig, axes, runs, title, subtitle, path):
    for ax in np.ravel(axes):
        if ax.get_visible():
            ax.set_xlabel("iteration")
    fig.suptitle(title, x=0.012, y=0.995, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.012, 0.952, subtitle, ha="left", fontsize=9.5, color=INK_2)
    if len(runs) > 1:
        handles = [plt.Line2D([], [], color=r["color"], lw=2.0) for r in runs]
        fig.legend(handles, [r["label"] for r in runs], loc="upper right", ncol=len(runs), bbox_to_anchor=(0.995, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("  wrote", path)


def page_training(runs, a, out):
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    draw(ax[0, 0], runs, "reward", "Mean reward", "reward per episode", a)
    if draw(ax[0, 1], runs, "ep_len", "Episode length", "seconds", a, transform=lambda y, r: y * r["dt"]):
        ref_line(ax[0, 1], runs[0]["ep_max_s"], f"full episode {runs[0]['ep_max_s']:.0f} s")
    draw(ax[0, 2], runs, "collapsed", "Collapses", "robots collapsed per reset", a)
    draw(ax[1, 0], runs, "noise", "Policy noise (exploration)", "action std", a)
    draw(ax[1, 1], runs, "value_loss", "Value loss", "loss (log)", a, logy=True)
    draw(ax[1, 2], runs, "lr", "Learning rate", "lr (log)", 0.0, logy=True)
    finish(fig, ax, runs, "Training health",
           "Healthy: reward and episode length rise and level off, collapses fall to ~0, noise shrinks slowly.",
           os.path.join(out, "01_training.png"))


def page_task(runs, a, out, radius):
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    if draw(ax[0, 0], runs, "speed", "Forward speed", "m/s", a):
        lo, hi = runs[0]["speed_cmd"]
        ax[0, 0].axhspan(lo, hi, color=GRID, alpha=0.6, zorder=0, lw=0)
        ref_line(ax[0, 0], (lo + hi) / 2, f"commanded {lo:g} to {hi:g} m/s (mean {(lo + hi) / 2:g})")
    draw(ax[0, 1], runs, "vel_err", "Speed error", "|speed - command| [m/s]", a)
    lap = 2 * np.pi * radius

    def laps(y, run):
        ep = run["data"].get(TAGS["ep_len"])
        if ep is None:
            return y * run["ep_max_s"] / lap
        return y * np.interp(run["data"][TAGS["speed"]][0], ep[0], ep[1]) * run["dt"] / lap

    draw(ax[0, 2], runs, "speed", f"Laps per episode (circle {lap:.1f} m)", "laps", a, transform=laps)
    draw(ax[1, 0], runs, "height_err", "Body height error", "cm off standing height", a)
    if draw(ax[1, 1], runs, "contact", "Gait match", "share of steps on the clock", a):
        ax[1, 1].set_ylim(0, 1.02)
        ref_line(ax[1, 1], 0.5, "random = 0.5", ":")
    draw(ax[1, 2], runs, "clear_err", "Swing clearance error", "cm short of swing height", a)
    finish(fig, ax, runs, "Doing the job",
           "Speed inside the grey band, laps going up, height error and clearance error going down, gait match near 1.",
           os.path.join(out, "02_task.png"))


def page_motor(runs, a, out):
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    if draw(ax[0, 0], runs, "torque_rms", "RMS torque (worst joint)", "Nm", a):
        ref_line(ax[0, 0], 5.46, "continuous limit 5.46 Nm")
    if draw(ax[0, 1], runs, "impact", "Touchdown impact", "peak force [dog body weights]", a):
        ref_line(ax[0, 1], 3.0, "target 3 BW")
    draw(ax[1, 0], runs, "power", "Mean electrical power", "W", a)
    draw(ax[1, 1], runs, "cot", "Cost of transport", "energy / (weight x distance)", a)
    finish(fig, ax, runs, "Motor limits and energy",
           "Stay under the dashed lines. Power and cost of transport: lower is better, once the leg is moving.",
           os.path.join(out, "03_motor_limits.png"))


def page_rewards(runs, a, out):
    terms = sorted({t for r in runs for t in r["data"] if t.startswith("Episode_Reward/")})
    if not terms:
        print("  (no Episode_Reward/* tags, skipping 04_rewards.png)")
        return
    cols = 3
    rows = int(np.ceil(len(terms) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(14, 2.9 * rows + 0.9), squeeze=False)
    for i, tag in enumerate(terms):
        a_ = ax[i // cols, i % cols]
        draw(a_, runs, tag, tag.split("/", 1)[1], "per second", a)
        a_.axhline(0, color=INK_3, lw=0.8)
    for j in range(len(terms), rows * cols):
        ax[j // cols, j % cols].set_visible(False)
    finish(fig, ax, runs, "Reward terms",
           "Each term on its own scale. Penalties (below 0) should shrink toward 0; the big positive terms should rise.",
           os.path.join(out, "04_rewards.png"))


# ------------------------------------------------------------------------------------------------- summary
def summary(runs, out):
    rows = [("metric", "target")] + [(k, f"{d} {v:g}") for k, (d, v) in TARGETS.items()] + [("tune score", "lower")]
    cols = []
    for run in runs:
        m = {}
        for k in TARGETS:
            src = "ep_len" if k == "ep_len_s" else k
            if TAGS[src] in run["data"]:
                v = float(np.mean(run["data"][TAGS[src]][1][-20:]))
                m[k] = v * run["dt"] if k == "ep_len_s" else v
        col = [run["label"]]
        for k, (d, v) in TARGETS.items():
            if k not in m:
                col.append("-")
            else:
                ok = m[k] < v if d == "<" else m[k] > v
                col.append(f"{m[k]:.3g} {'ok' if ok else 'MISS'}")
        col.append(f"{tune_score(run):.2f}")
        cols.append(col)
    w0 = max(len(r[0]) for r in rows) + 2
    w1 = max(len(r[1]) for r in rows) + 2
    wc = [max(len(c) for c in col) + 2 for col in cols]
    lines = []
    for i, (name, tgt) in enumerate(rows):
        lines.append(name.ljust(w0) + tgt.ljust(w1) + "".join(col[i].ljust(w) for col, w in zip(cols, wc)))
    text = "last 20 iterations, averaged\n\n" + "\n".join(lines)
    print("\n" + text + "\n")
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(text + "\n")


def tune_score(run):
    """Same score as tune_simple.py, so you can compare a hand-run with the Optuna trials."""
    g = lambda k, d=0.0: float(np.mean(run["data"][TAGS[k]][1][-10:])) if TAGS[k] in run["data"] else d
    return (g("height_err", 10) + 20 * (1 - g("contact")) + g("clear_err", 10) + 10 * g("vel_err")
            + 3 * max(0, g("torque_rms") - 5.46) + 2 * max(0, g("impact") - 3.0) + g("power") / 20)


# ------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None, help="run folder names or paths (default: latest), max 3")
    ap.add_argument("--labels", nargs="*", default=None, help="legend names for --runs")
    ap.add_argument("--smooth", type=float, default=0.9, help="0..0.99, like TensorBoard's slider")
    ap.add_argument("--radius", type=float, default=1.5, help="boom radius [m], for laps")
    ap.add_argument("--out", default=None, help="output folder (default logs/plots/<first run name>)")
    args = ap.parse_args()

    names = args.runs or [None]
    if len(names) > 3:
        raise SystemExit("compare at most 3 runs at a time (more colours stop being easy to tell apart)")
    runs = []
    for i, name in enumerate(names):
        d = find_run(name)
        cfg = read_env_cfg(d)
        sim = cfg.get("sim") or {}
        dt = float(sim.get("dt", 1 / 200)) * int(cfg.get("decimation", 4))
        speed_cmd = cfg.get("fwd_speed_range") or [0.1, 0.4]
        label = args.labels[i] if args.labels and i < len(args.labels) else os.path.basename(os.path.normpath(d))
        print(f"reading {d}")
        runs.append({"dir": d, "data": read_scalars(d), "label": label, "color": RUN_COLORS[i], "dt": dt,
                     "ep_max_s": float(cfg.get("episode_length_s", 10.0)), "speed_cmd": [float(v) for v in speed_cmd]})

    out = args.out or os.path.join(OUT_ROOT, re.sub(r"[^\w.-]", "_", os.path.basename(os.path.normpath(runs[0]["dir"]))))
    os.makedirs(out, exist_ok=True)
    page_training(runs, args.smooth, out)
    page_task(runs, args.smooth, out, args.radius)
    page_motor(runs, args.smooth, out)
    page_rewards(runs, args.smooth, out)
    summary(runs, out)


if __name__ == "__main__":
    main()
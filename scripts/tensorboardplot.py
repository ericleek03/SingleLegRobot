import os,glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs = sorted(glob.glob(os.path.join(ROOT, "logs/rsl_rl/single_leg_direct/*/")))
log_dir = runs[-1]  # most recent run
print("run:", log_dir)
event_acc = EventAccumulator(log_dir, size_guidance={"scalars": 0})  # 0 = load all points
event_acc.Reload()
available = set(event_acc.Tags()["scalars"])

# output folders inside the run directory
OUT_DIR = os.path.join(log_dir, "analysis")
CSV_DIR = os.path.join(OUT_DIR, "csv")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# f(t) = f(t-1) * w + (1-w) * data[t]
def smooth(values, weight=0.9):
    last = values[0] # first value in plot
    smoothed = list()
    for point in values:
        smoothed_val = last*weight + (1-weight)*point # smooth funtion
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

#print("Available Data Tags:", event_acc.Tags())
learning_tags = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode_Termination/collapsed",
]
reward_tags = [
    "Episode_Reward/track_vel",
    "Episode_Reward/height",
    "Episode_Reward/gait",
    "Episode_Reward/clearance",
    "Episode_Reward/energy",
    "Episode_Reward/impact",
    "Episode_Reward/foot_slip",
    "Episode_Reward/action_rate",
    "Episode_Reward/collapse",
]
task_tags = [
    "Episode/mean_speed_ms",
    "Episode/vel_error_ms",
    "Episode/body_height_error_cm",
    "Episode/contact_match",
    "Episode/clearance_error_cm",
]
physical_tags = [
    "Episode/torque_rms_max_Nm",
    "Episode/mean_power_W",
    "Episode/cost_of_transport",
    "Episode/impact_peak_bw",
    "Episode/contact_match",
]

# 5. PPO health (debugging)
ppo_tags = [
    "Policy/mean_std",
    "Loss/value",
    "Loss/surrogate",
    "Loss/entropy",
    "Loss/learning_rate",
]

# Short list for a report
report_tags = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode/mean_speed_ms",
    "Episode/torque_rms_max_Nm",
]

def export_tags(tags, group_name, show=False):
    """Save one CSV per tag, one combined CSV per group, and one PNG per tag."""
    merged = None
    for tag in tags:
        if tag not in available:
            print(f"  [skip] {tag} not found in this run")
            continue
        ev = event_acc.Scalars(tag)
        steps = [e.step for e in ev]
        vals = [e.value for e in ev]
        safe = tag.replace("/", "_")

        # per-tag CSV (raw + smoothed)
        df = pd.DataFrame({"step": steps, "value": vals, "smoothed": smooth(vals)})
        df.to_csv(os.path.join(CSV_DIR, f"{safe}.csv"), index=False)

        # build combined group table (one column per tag, aligned on step)
        col = df[["step", "value"]].rename(columns={"value": tag})
        merged = col if merged is None else merged.merge(col, on="step", how="outer")

        # plot: raw faint, smoothed solid
        fig, ax = plt.subplots()
        ax.plot(steps, vals, alpha=0.3, label="raw")
        ax.plot(steps, smooth(vals), label="smoothed")
        ax.set_title(tag)
        ax.set_xlabel("iteration")
        ax.set_ylabel(tag.split("/")[-1])
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"{safe}.png"), dpi=150)
        if not show:
            plt.close(fig)

    if merged is not None:
        merged.sort_values("step").to_csv(os.path.join(CSV_DIR, f"{group_name}.csv"), index=False)
    print(f"saved {group_name}")


groups = {
    "learning": learning_tags,
    "reward": reward_tags,
    "task": task_tags,
    "physical": physical_tags,
    "ppo": ppo_tags,
    "report": report_tags,
}
for name, tags in groups.items():
    export_tags(tags, name)

print("outputs in:", OUT_DIR)

# Had claude help clean up and organize the file to get my plots and save csv
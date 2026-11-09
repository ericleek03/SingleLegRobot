import os,glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs = sorted(glob.glob(os.path.join(ROOT, "logs/rsl_rl/single_leg_direct/*/")))
log_dir = runs[-1]
print("run:", log_dir)
event_acc = EventAccumulator(log_dir)
event_acc.Reload()
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
for tag in report_tags:
    ev = event_acc.Scalars(tag)
    steps= [e.step for e in ev]
    vals = [e.value for e in ev]
    plt.figure()
    plt.plot(smooth(steps), smooth(vals))
    plt.ylabel(tag)
    plt.xlabel("iteration")

    
plt.show()
    
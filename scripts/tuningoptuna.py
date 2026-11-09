import argparse
import glob
import json
import math
import os
import re
import signal
import subprocess
import sys
import time

import optuna
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TASK = "Template-Singlelegrobot-Direct-v0"
TRAIN = "scripts/rsl_rl/train.py"
LOG_ROOT = "logs/rsl_rl/single_leg_direct"
ITERATION = 300
NUM_ENVS = 512


# Optuna changes the cfg name 
# Penalties

SEARCH = {
    "height_sigma": (0.01, 0.04, False),
    "rew_gait": (0.5, 5.0, False),
    "rew_clearance": (0.0, 3.0, False),
    "rew_energy": (0.005, 0.05, True), # penalty is true
    "rew_impact": (0.01, 1.0, True), # our range 
    "rew_foot_slip": (2.0, 50.0, True),
    "rew_action_rate": (0.002, 0.05, True),
}

PENALTIES = {"rew_energy", "rew_impact", "rew_foot_slip", "rew_action_rate"}

TAGS = {"height_err": "Episode/body_height_error_cm",
        "contact": "Episode/contact_match",
        "clear_err": "Episode/clearance_error_cm",
        "torque_rms": "Episode/torque_rms_max_Nm",
        "impact": "Episode/impact_peak_bw",
        "power": "Episode/mean_power_W",}

def score(m):
    if len(m) < len(TAGS):
        return 100.0 # training failed or metrics missing
    return (m["height_err"] + 20 * (1-m["contact"]) + m["clear_err"] + 3 * max(0, m["torque_rms"] - 5.46) + 2*max(0, m["impact"] - 3.0) + m["power"] / 20)

def read_metrics(run_dir):
    acc = EventAccumulator(run_dir)
    acc.Reload()  
    found = acc.Tags()["scalers"]
    m = {}
    for key, tag in TAGS.items():
        if tag in found:
            values = [e.value for e in acc.Scalers(tag)[-10:]]
            m[key] = sum(values) / len(values)
        return m

def objective(trial):
    overrides = []
    for name, (lo, hi, log) in SEARCH.items():
        value = trial.suggest_float(name,lo,hi,log=log)
        if name in PENALTIES:
            value = -value
            overrides.append(f"env.{name} = {value}")

    run_name = f"optuna_t{trial.number:03d}"
    cmd = [sys.executable, TRAIN, f"--task={TASK}", "--headless", f"--max_iterations={ITERATION}",
           f"--num_envs={NUM_ENVS}", f"--run_name={run_name}"] + overrides
    print(f"\n[trial {trial.number}] {' '.join(overrides)}")
    os.makedirs("logs/optuna", exist_ok=True)
    with open(f"logs/optuna/trial_{trial.number:03d}.log", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT) 

    runs = glob.glob(os.path.join(LOG_ROOT, f"*_{run_name}"))
    m = read_metrics(max(runs, key=os.path.getmtime)) if runs else {}
    for k, v in m.items():
        trial.set_user_attr(k, round(v, 3))
    s = score(m)
    print(f"[trial {trial.number}] score {s:.2f}  " + "  ".join(f"{k} {v:.2f}" for k, v in m.items()))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type = int, default = 30)
    args = ap.parse_args()

    os.makedirs("logs/optuna/plots", exist_ok=True)
    study = optuna.create_study(study_name="trot_simple", storage="sqlite:///logs/optuna/trot_simple.db",
                                load_if_exists=True, direction="minimize",
                                sampler=optuna.samplers.RandomSampler(seed=42))
    study.optimize(objective, n_trials=args.trials)
 
    best = study.best_trial
    print(f"\nbest trial {best.number}: score {best.value:.2f}")
    print("measurements:", best.user_attrs)
    print("settings:")
    for name, value in best.params.items():
        print(f"   env.{name}={-value if name in PENALTIES else value:.5g}")
    if len(study.trials) >= 3:
        optuna.visualization.plot_optimization_history(study).write_html("logs/optuna/plots/history.html")
        optuna.visualization.plot_param_importances(study).write_html("logs/optuna/plots/importance.html")
        optuna.visualization.plot_slice(study).write_html("logs/optuna/plots/slice.html")
        optuna.visualization.plot_pareto_front(study).write_html("logs/optuna/plots/pareto_front.html")
        optuna.visualization.plot_contour(study).write_html("logs/optuna/plots/contour.html")
        optuna.visualization.plot_parallel_coordinate(study).write_html("logs/optuna/plots/parallel_coordinate.html")
        print("graphs: logs/optuna/plots/history.html, importance.html, slice.html")

    df = study.trails_dataframe()
    df.to_csv("hyperparameter_trials_data.csv", index=False)
 
if __name__ == "__main__":
    main()  
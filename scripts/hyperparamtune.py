import optuna 
import optuna.visualization as vis
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import glob
import subprocess
import sys
#Event accumulator reads binary tensorboard files from the disk, allowing us to extract scalar metrics to be used in optuna trials

TASK = "Template-Singlelegrobot-Direct-v0"
TRAIN = "scripts/rsl_rl/train.py"
LOG_ROOT = "logs/rsl_rl/single_leg_direct"
ITERATION = 200
NUM_ENVS = 1024
SEED = 42
STUDY = "trot_v5"
DB = "sqlite:///logs/optuna/trot_v5.db"
CHECK_EVERY = 60 # seconds
TRIALS = 50
PLOTS = "logs/optuna/plots"
OBJECTIVE_TAG = "Episode/contact_match"  # the one metric the tuner optimizes

# selected a range from our initial test of the system without tuning
SEARCH = {
    # positive rewards
    #"rew_height": (0.5, 6.0, False),
    "rew_gait": (1, 8.0, False),
    "rew_clearance": (0.5, 6.0, False),
    "rew_track_vel": (1.0, 8.0, False),
    # penalties (entered positive, made negative in objective)
    "rew_energy": (0.0005, 0.02, True),
    "rew_impact": (0.005, 1.0, True),
    "rew_foot_slip": (1.0, 40.0, True),
    "rew_action_rate": (0.001, 0.05, True),
    #"rew_collapse": (1.0, 50.0, True),
    # reward shape
    "height_sigma": (0.02, 0.08, True),
    "clearance_sigma": (0.02, 0.1, True),
    "vel_sigma": (0.05, 0.35, True),
    #"impact_threshold_bw": (1.5, 5.0, False),
}
PENALTIES = {"rew_energy", "rew_impact", "rew_foot_slip", "rew_action_rate", "rew_collapse"}

TAGS = {"height_err": "Episode/body_height_error_cm",
        "contact": "Episode/contact_match",
        "clear_err": "Episode/clearance_error_cm",
        "torque_rms": "Episode/torque_rms_max_Nm",
        "impact": "Episode/impact_peak_bw",
        "power": "Episode/mean_power_W",}

FIXED = {
    "rew_height": 2.0,
    "rew_collapse": -5.0,
    "impact_threshold_bw": 2.5,
    "height_mode": "floor",
}
def read_metrics(run_dir):
    event_acc = EventAccumulator(run_dir)
    event_acc.Reload()
    metrics = {}
    for key, tag in TAGS.items():
        values = [e.value for e in event_acc.Scalars(tag)[-10:]] # average of 10 to reduce noise
        metrics[key] = sum(values) / len(values)
    return metrics

def find_best_score(scalar_events, window_percentage = 0.1):
    #using a time weight windowed average to calculate best score
    #it averages the last 10% of the total training steps 
    # if the leg falls over the calculation penalizes the instability
    # translates smoothly to physical hardware, consistent, steady state standing
    if not scalar_events:
        return float("-inf")

    values = [event.value for event in scalar_events]
    total_steps = len(values)
    #Determines how many steps represents the final tail 
    window_size = max(1, int(total_steps * window_percentage))
    # slice the last N items (10%)
    final_window_values = values[-window_size:]
    #calculate final score
    final_score = sum(final_window_values)/len(final_window_values)

    return final_score

def objective(trial):
    overrides = []
    for name, (lo, hi, log) in SEARCH.items():
        value = trial.suggest_float(name, lo, hi, log=log)
        if name in PENALTIES:
            value = -value
        overrides.append(f"env.{name}={value}")
    
    overrides += [f"env.{name}={value}" for name, value in FIXED.items()]

    #my model 
    # training model
    run_name = f"optuna_v2_t{trial.number:03d}"
    cmd = [sys.executable, TRAIN, f"--task={TASK}", "--headless",
           f"--max_iterations={ITERATION}", f"--num_envs={NUM_ENVS}",
           f"--run_name={run_name}"] + overrides
    with open(f"logs/optuna/v2_trial_{trial.number:03d}.log", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT) # runs training and logs the data into v2_trial
    
    # given the score of the model
    runs = sorted(glob.glob(f"{LOG_ROOT}/*_{run_name}"))
    if not runs:
        raise RuntimeError(f"trial {trial.number} produced no run folder, see its log")
    event_acc = EventAccumulator(runs[-1]) # takes our data and inputs it into our bestscore algorithm
    event_acc.Reload()
    return find_best_score(event_acc.Scalars(OBJECTIVE_TAG), window_percentage=0.1)



if __name__ == "__main__":

    study = optuna.create_study(study_name = STUDY,
                                storage = DB,
                                direction = "maximize",
                                load_if_exists=True,
                                sampler = optuna.samplers.RandomSampler(seed=SEED)) # ran TPE first 
    #Run optimization
    print("Start Optimization")
    study.optimize(objective, n_trials=TRIALS, catch=(RuntimeError, KeyError))
    # results
    study.trials_dataframe().to_csv("logs/optuna/tuned_data.csv", index=False)
    print(f"Best value: {study.best_value} (params: {study.best_params})\n")
    print("Best Configuration Parameters:")
    for key, value in study.best_params.items():
        print(f" {key}: {value}")
    
    top = list(optuna.importance.get_param_importances(study))
    vis.plot_optimization_history(study).write_html("optimization_history.html")
    vis.plot_contour(study, params=["rew_gait", "rew_foot_slip"]).write_html("contour.html")
    vis.plot_slice(study, params=top[:6]).write_html("slice.html")
    vis.plot_param_importances(study).write_html("param_importances.html")
    vis.plot_parallel_coordinate(study, params=top[:6]).write_html("parallel_coordinates.html")
    vis.plot_optimization_history(study, target=lambda t: t.user_attrs["torque_rms"],
                                     target_name="torque RMS (Nm)").write_html("torque_rms_optimization_history.html")
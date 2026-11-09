import optuna 
import optuna.visualization as vis
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline
from tuningoptuna import SEARCH, PENALTIES, TASK

SEED = 42
STUDY = "trot_v2"
DB = "sqlite:///logs/optuna/trot_v2.db"
CHECK_EVERY = 60

def model(trial):
    out = []
    for name, (lo, hi, log) in SEARCH.items():
        v = trial.suggest_float(name, lo, hi, log = log)
        out.append(f"env.{name}={-v if name in PENALITIES else v}")
    return out  

def objective(trial):
    train_model
    for epoch in range(10):


if __name__ = "__main__":
    os.makedirs("logs")
study = optuna.create_study(
    direction = "maximize",
    sampler = optuna.sampler.TPESampler(seed=SEED),
    prune = optuna.pruners.MedianPruner(),

)
study.optimize(objective, n_trials=30, timeout=300)

optuna.visualization.plot_param_importances(study, target=lambda t: t.duration.t)
optuna.visualization.plot_pareto_front()
optuna.visualization.plot_contour

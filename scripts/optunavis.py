import optuna
import optuna.visualization as vis

study = optuna.load_study(
    study_name = "trot_v5",
    storage = "sqlite:///logs/optuna/trot_v5.db"
)

top = list(optuna.importance.get_param_importances(study))
vis.plot_optimization_history(study).write_html("logs/optuna/plots/optimization_history.html")
vis.plot_contour(study, params=["rew_gait", "rew_foot_slip"]).write_html("logs/optuna/plots/contour.html")
vis.plot_slice(study, params=top[:6]).write_html("logs/optuna/plots/slice.html")
vis.plot_param_importances(study).write_html("logs/optuna/plots/param_importances.html")
vis.plot_parallel_coordinate(study, params=top[:6]).write_html("logs/optuna/plots/parallel_coordinates.html")
#vis.plot_optimization_history(study, target=lambda t: t.user_attrs["torque_rms"],
 #                                 target_name="torque RMS (Nm)").write_html("torque_rms_optimization_history.html")


"""Single-leg implicit-drive bench test. Run with the project's Isaac Lab Python.
Requires the installed, runnable SingleLegRobot task (see zero_agent.py).
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--task", default="Template-Singlelegrobot-Direct-v0")
p.add_argument("--joint", choices=["hip_joint", "knee_joint"], required=True)
p.add_argument("--fixture", choices=["unloaded", "stance"], required=True)
p.add_argument("--kp", type=float, default=42.)
p.add_argument("--kd", type=float, default=3.5)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", type=Path, default=Path("pd_step_results"))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
launcher = AppLauncher(args)
app = launcher.app


def run():
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    import SingleLegRobot.tasks  # registers the external project
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.seed = args.seed
    # Keep gains deterministic for the isolated comparison; the training
    # environment randomizes them by default on startup.
    if getattr(cfg.events, "motor_gains", None) is not None:
        cfg.events.motor_gains = None
    drive = cfg.robot_cfg.actuators["leg_motors"]
    # Override this motor only; keep the held joint at the reference gains.
    drive.stiffness = {j: args.kp if j == args.joint else 42. for j in ("hip_joint","knee_joint")}
    drive.damping = {j: args.kd if j == args.joint else 3.5 for j in ("hip_joint","knee_joint")}
    if args.fixture == "unloaded":
        x,y,z = cfg.robot_cfg.init_state.pos
        cfg.robot_cfg.init_state.pos = (x,y,z+1.5)
    env = gym.make(args.task, cfg=cfg)
    try:
        base = env.unwrapped
        env.reset(seed=args.seed)
        robot = base.robot
        joints = ("hip_joint","knee_joint")
        ids = [robot.find_joints(j)[0][0] for j in joints]
        z_id = robot.find_joints("pz_z")[0][0]
        sensor = base.scene.sensors["contact_sensor"]
        foot_id = sensor.find_bodies(cfg.foot_body)[0][0]
        tested = joints.index(args.joint)
        nominal = robot.data.default_joint_pos[:,ids].clone()
        limits = robot.data.joint_pos_limits[0,ids[tested]]
        target = float(nominal[0,tested])+.10
        if not float(limits[0]) <= target <= float(limits[1]):
            raise ValueError(f"Requested target {target} outside joint limits {limits}.")
        dt = float(base.physics_dt)
        out = args.out / (datetime.now().strftime("%Y%m%d_%H%M%S_%f")+f"_{args.fixture}_{args.joint}")
        out.mkdir(parents=True)
        meta = dict(task=args.task, fixture=args.fixture, joint=args.joint, kp=args.kp,kd=args.kd,
                    other_joint_kp=42., other_joint_kd=3.5, seed=args.seed,dt_s=dt,
                    armature=robot.data.joint_armature[0,ids].tolist(),
                    masses_kg=robot.root_physx_view.get_masses()[0].tolist(),
                    body_order=robot.body_names, runtime_stiffness=robot.data.joint_stiffness[0,ids].tolist(),
                    runtime_damping=robot.data.joint_damping[0,ids].tolist(), joint_order=joints,
                    nominal_rad=nominal[0].tolist(),support="off; no gait clock or task support wrench applied")
        (out/"settings.json").write_text(json.dumps(meta,indent=2))
        rows=[]
        # Bypass env.step(): no policy, clock-based support, pushes or auto-resets.
        # Physics and actuator implementation are still those of the project.
        with torch.inference_mode():
            for phase,sec in (("hold",2.),("step",1.),("return",1.)):
                cmd=nominal.clone()
                if phase=="step":cmd[0,tested]+=.10
                for _ in range(round(sec/dt)):
                    if not app.is_running():raise RuntimeError("Simulation closed before test finished.")
                    robot.set_joint_position_target(cmd,joint_ids=ids)
                    robot.set_joint_velocity_target(torch.zeros_like(cmd),joint_ids=ids)
                    base.scene.write_data_to_sim()
                    base.sim.step()
                    base.scene.update(dt)
                    data=robot.data
                    f=sensor.data.net_forces_w[0,foot_id]
                    row=dict(time_s=(len(rows)+1)*dt,phase=phase,fixture=args.fixture,
                             foot_fz_N=f[2].item(),contact=int(f.norm().item()>cfg.contact_threshold),
                             carriage_z_m=data.joint_pos[0,z_id].item(),carriage_vz_m_s=data.joint_vel[0,z_id].item())
                    for i,j in enumerate(joints):
                        k=j.split("_")[0]
                        row.update({k+"_target_rad":cmd[0,i].item(),k+"_angle_rad":data.joint_pos[0,ids[i]].item(),
                                    k+"_speed_rad_s":data.joint_vel[0,ids[i]].item(),
                                    k+"_effort_est_Nm":data.applied_torque[0,ids[i]].item()})
                    rows.append(row)
        with (out/"trace.csv").open("w",newline="") as fp:
            w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        print("Results:", out.resolve())
        tail=[r for r in rows if r["phase"]=="hold"][-max(1,round(.2/dt)):]
        print("Last 0.2 s of hold: contact fraction",sum(r["contact"] for r in tail)/len(tail),
              "max abs joint speed",max(abs(r[j+"_speed_rad_s"]) for r in tail for j in ("hip","knee")))
        print("Verify initial contact and settling before comparing. Unloaded slider may rest on its lower stop.")
    finally:
        env.close()


try:
    run()
finally:
    app.close()

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One leg of a trotting dog. Works on three rigs, detected from the URDF:
    boom      radial boom: pz_yaw (swing around the post) + pz_pitch (tilt) + pz_level (parallelogram linkage)
    rails     straight rails: pz_x (horizontal) + pz_z (vertical)
    vertical  vertical slider: pz_z only (no forward motion)
Carry its share of the body in stance and push it forward at the commanded speed, swing on time with the foot
lifted, touch down softly, stay within the motor's limits, use little energy. See the cfg."""

from __future__ import annotations

import math
import torch
import xml.etree.ElementTree as ET
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .singlelegrobot_env_cfg import LEG_MASS, SinglelegrobotEnvCfg


TIP_BODIES = ["mount_link", "base_link", "hip_link", "knee_link", "foot_link"]   # what rides at the leg mount


def read_rig(urdf_path: str) -> dict:
    """Which rig the URDF describes, and (boom) how much static load the boom adds at the leg."""
    root = ET.parse(urdf_path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    if "pz_yaw" in joints:
        radius = abs(float(joints["pz_level"].find("origin").get("xyz").split()[1]))
        boom = next(link for link in root.findall("link") if link.get("name") == "boom_link").find("inertial")
        mass = float(boom.find("mass").get("value"))
        com_y = float(boom.find("origin").get("xyz").split()[1])
        return {"type": "boom", "radius": radius, "extra_mass": mass * (-com_y) / radius}
    if "pz_x" in joints:
        return {"type": "rails", "radius": 0.0, "extra_mass": 0.0}
    return {"type": "vertical", "radius": 0.0, "extra_mass": 0.0}


class SinglelegrobotEnv(DirectRLEnv):
    cfg: SinglelegrobotEnvCfg

    def __init__(self, cfg: SinglelegrobotEnvCfg, render_mode: str | None = None, **kwargs):
        # carriage mass = this leg's share of the dog, minus the leg and (boom) the boom's weight felt at the tip.
        # Set before the event manager is built.
        self.rig = read_rig(cfg.robot_cfg.spawn.asset_path)
        lo, hi = cfg.payload_range
        if cfg.mount_mass is None:      # dog-load test: the leg carries exactly its share of the dog
            share = cfg.body_mass * cfg.load_share
            base = LEG_MASS + self.rig["extra_mass"]
            cfg.events.carriage_mass.params["mass_distribution_params"] = (lo * share - base, hi * share - base)
        else:                           # real-rig match: the mount hardware as it is (counterweight lightens it)
            cfg.events.carriage_mass.params["mass_distribution_params"] = (lo * cfg.mount_mass, hi * cfg.mount_mass)
        super().__init__(cfg, render_mode, **kwargs)

        # ---- indices ----
        self.dof_idx, dof_names = self.robot.find_joints(self.cfg.dof_names)
        if self.rig["type"] == "boom":
            self.pitch_idx = self.robot.find_joints("pz_pitch")[0][0]
            self.level_idx = self.robot.find_joints("pz_level")[0][0]
            self.push_dof, self.push_gain = self.pitch_idx, -1.0 / self.rig["radius"]   # +pitch lowers the tip
        else:
            self.push_dof, self.push_gain = self.robot.find_joints("pz_z")[0][0], 1.0
        self.foot_id = self._contact.find_bodies(self.cfg.foot_body)[0][0]
        self.foot_body_idx = self.robot.find_bodies(self.cfg.foot_body)[0][0]
        self.mount_body_idx = self.robot.find_bodies("mount_link")[0][0]
        tip_ids = self.robot.find_bodies(TIP_BODIES)[0]

        # static load at the leg (after the mass randomisation at startup) and the dog's weight
        masses = self.robot.root_physx_view.get_masses().to(self.device)        # (num_envs, num_bodies)
        self.moving_weight = (masses[:, tip_ids].sum(dim=1) + self.rig["extra_mass"]) * 9.81
        self.dog_weight = self.cfg.body_mass * 9.81

        n = self.num_envs
        self.sole_offset = torch.tensor(self.cfg.sole_offset, device=self.device).repeat(n, 1)
        self.actions = torch.zeros(n, len(self.dof_idx), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        # gait clock + commands
        self.phase = torch.zeros(n, device=self.device)                 # 0..1 per step cycle, stance first
        self.gait_freq = torch.zeros(n, device=self.device)
        self.swing_h = torch.zeros(n, device=self.device)
        self.fwd_cmd = torch.zeros(n, device=self.device)
        self.prev_stance = torch.ones(n, dtype=torch.bool, device=self.device)
        self.push_timer = torch.zeros(n, device=self.device)
        # energy / torque bookkeeping (filled every physics step in _apply_action)
        self._step_energy = torch.zeros(n, device=self.device)
        self._energy_sum = torch.zeros(n, device=self.device)
        self._tau_sq_sum = torch.zeros(n, len(self.dof_idx), device=self.device)
        # metrics for TensorBoard / Optuna
        self._height_err_sum = torch.zeros(n, device=self.device)
        self._gait_match_sum = torch.zeros(n, device=self.device)
        self._swing_peak = torch.zeros(n, device=self.device)
        self._clear_short_sum = torch.zeros(n, device=self.device)
        self._swings = torch.zeros(n, device=self.device)
        self._impact_peak = torch.zeros(n, device=self.device)
        self._vel_err_sum = torch.zeros(n, device=self.device)
        self._distance_sum = torch.zeros(n, device=self.device)
        self._episode_sums: dict[str, torch.Tensor] = {}
        self._debug_steps = 0

        print(f"[INFO] Leg joints controlled by the policy: {dof_names}")
        print(f"[INFO] Links: {self.robot.body_names}")
        target = (f"target {self.cfg.body_mass * self.cfg.load_share:.2f} kg = {self.cfg.load_share} x "
                  f"{self.cfg.body_mass} kg dog" if self.cfg.mount_mass is None else f"real mount {self.cfg.mount_mass} kg")
        print(f"[INFO] Load at the leg mount: {self.moving_weight.mean().item() / 9.81:.2f} kg ({target})")
        print(f"[INFO] Rig: {self.rig['type']}" + (f" (radius {self.rig['radius']} m, boom adds "
              f"{self.rig['extra_mass']:.2f} kg)" if self.rig["type"] == "boom" else "")
              + f" | support: {self.cfg.support_mode} | speed command {self.cfg.fwd_speed_range} m/s")

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact = ContactSensor(self.cfg.contact_sensor)
        # "multiply": the foot's (randomised) friction decides, since the ground's is 1.0
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                friction_combine_mode="multiply", restitution_combine_mode="min")),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---------------------------------------------------------------- readings
    # The leg mount's position/velocity in the world works the same on every rig.
    def mount_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.robot.data
        pos = d.body_link_pos_w[:, self.mount_body_idx] - self.scene.env_origins
        return pos, d.body_link_lin_vel_w[:, self.mount_body_idx]

    def forward_dir(self) -> torch.Tensor:
        """Unit vector of 'forward' at the leg: +x on rails, along the circle (tangent) on the boom."""
        if self.rig["type"] == "boom":
            r = self.mount_state()[0][:, :2]
            r = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            return torch.stack((-r[:, 1], r[:, 0], torch.zeros_like(r[:, 0])), dim=-1)    # z x r: +x at the start
        fwd = torch.zeros(self.num_envs, 3, device=self.device)
        fwd[:, 0] = 1.0
        return fwd

    def slider_z(self) -> torch.Tensor:
        """Body (leg mount) height relative to its start height [m]."""
        return self.mount_state()[0][:, 2] - self.cfg.start_height

    def slider_vz(self) -> torch.Tensor:
        return self.mount_state()[1][:, 2]

    def slider_vx(self) -> torch.Tensor:
        """Forward speed of the body [m/s] (0 on the vertical slider)."""
        if self.rig["type"] == "vertical":
            return torch.zeros(self.num_envs, device=self.device)
        return (self.mount_state()[1] * self.forward_dir()).sum(dim=-1)

    def foot_force(self) -> torch.Tensor:
        return self._contact.data.net_forces_w[:, self.foot_id].norm(dim=-1)

    def foot_contact(self) -> torch.Tensor:
        return self.foot_force() > self.cfg.contact_threshold

    def stance_phase(self) -> torch.Tensor:
        return self.phase < self.cfg.duty_factor

    def sole_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World position and velocity of the point where the foot ball touches the ground."""
        d = self.robot.data
        q = d.body_link_quat_w[:, self.foot_body_idx]
        r = quat_apply(q, self.sole_offset)                              # link origin -> ball centre
        r[:, 2] -= self.cfg.foot_radius                                  # ball centre -> bottom of the ball
        pos = d.body_link_pos_w[:, self.foot_body_idx] + r
        vel = d.body_link_lin_vel_w[:, self.foot_body_idx] + torch.cross(
            d.body_link_ang_vel_w[:, self.foot_body_idx], r, dim=-1)
        return pos, vel

    def motor_power(self) -> torch.Tensor:
        """Battery power drawn by the two leg motors [W]: mechanical power + copper heat."""
        tau = self.robot.data.applied_torque[:, self.dof_idx]
        qd = self.robot.data.joint_vel[:, self.dof_idx]
        p = tau * qd + self.cfg.joule_coeff * tau**2
        if not self.cfg.regen:
            p = torch.clamp(p, min=0.0)
        return p.sum(dim=1)

    # ---------------------------------------------------------------- actions
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        self.actions = actions.clamp(-1.0, 1.0)
        if self.cfg.push_vel > 0.0:
            self._push_carriage()

    def _apply_action(self) -> None:
        targets = self.robot.data.default_joint_pos[:, self.dof_idx] + self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target(targets, joint_ids=self.dof_idx)
        if self.rig["type"] == "boom":      # parallelogram linkage: keep the leg upright while the boom tilts
            pitch = self.robot.data.joint_pos[:, self.pitch_idx]
            self.robot.set_joint_position_target(-pitch.unsqueeze(-1), joint_ids=[self.level_idx])
        self._apply_support()
        # every physics step (200 Hz): energy and torque^2 for the metrics
        tau = self.robot.data.applied_torque[:, self.dof_idx]
        self._step_energy += self.motor_power() * self.physics_dt
        self._tau_sq_sum += tau**2 * self.physics_dt

    def _apply_support(self):
        """Virtual 'other legs': an upward force on the carriage."""
        mode = self.cfg.support_mode
        if mode == "none":
            return
        if mode == "active":            # hold the body while THIS leg is in its swing phase (the clock, not the sensor)
            swing = (~self.stance_phase()).float()
            f = (self.moving_weight + self.cfg.support_stiffness * (self.cfg.stand_z - self.slider_z())
                 - self.cfg.support_damping * self.slider_vz())
            f = torch.clamp(f, min=0.0) * swing
        elif mode == "passive":         # counterweight / constant-force spring
            f = self.cfg.passive_fraction * self.moving_weight
        else:
            raise ValueError(f"support_mode must be 'active', 'passive' or 'none', not {mode!r}")
        forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        forces[:, 0, 2] = f
        if self.rig["type"] != "vertical" and mode == "active":
            # the other legs also keep the body moving at the commanded speed while THIS leg swings
            swing = (~self.stance_phase()).float()
            f_fwd = self.cfg.support_damping_x * (self.fwd_cmd - self.slider_vx()) * swing
            forces[:, 0, :] += f_fwd.unsqueeze(-1) * self.forward_dir()
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, body_ids=[self.mount_body_idx], is_global=True)

    def _push_carriage(self):
        """Random vertical kicks to the carriage (like a bump or a dropped payload)."""
        self.push_timer -= self.step_dt
        ids = (self.push_timer <= 0.0).nonzero(as_tuple=False).flatten()
        if len(ids) == 0:
            return
        pos = self.robot.data.joint_pos[ids].clone()
        vel = self.robot.data.joint_vel[ids].clone()
        kick = torch.empty(len(ids), device=self.device).uniform_(-self.cfg.push_vel, self.cfg.push_vel)
        vel[:, self.push_dof] += self.push_gain * kick
        self.robot.write_joint_state_to_sim(pos, vel, env_ids=ids)
        lo, hi = self.cfg.push_interval_s
        self.push_timer[ids] = torch.empty(len(ids), device=self.device).uniform_(lo, hi)

    # ---------------------------------------------------------------- observations (15)
    def _get_observations(self) -> dict:
        d = self.robot.data
        obs = torch.cat(
            (
                d.joint_pos[:, self.dof_idx] - d.default_joint_pos[:, self.dof_idx],   # 2 leg angles vs standing
                d.joint_vel[:, self.dof_idx],                                          # 2 leg speeds
                self.actions,                                                          # 2 last command
                torch.sin(2 * math.pi * self.phase).unsqueeze(-1),                     # 2 where in the step cycle
                torch.cos(2 * math.pi * self.phase).unsqueeze(-1),
                self.gait_freq.unsqueeze(-1),                                          # 1 command: steps per second
                self.swing_h.unsqueeze(-1),                                            # 1 command: foot height
                self.foot_contact().float().unsqueeze(-1),                             # 1 foot on the ground
                (self.slider_z() - self.cfg.stand_z).unsqueeze(-1),                    # 1 body height vs standing
                self.slider_vz().unsqueeze(-1),                                        # 1 body vertical speed
                self.fwd_cmd.unsqueeze(-1),                                            # 1 command: forward speed
                self.slider_vx().unsqueeze(-1),                                        # 1 forward speed (0 on slider)
            ),
            dim=-1,
        )
        return {"policy": obs}

    # ---------------------------------------------------------------- rewards
    def _get_rewards(self) -> torch.Tensor:
        z, contact, stance = self.slider_z(), self.foot_contact(), self.stance_phase()
        swing = ~stance
        sole_pos, sole_vel = self.sole_state()
        foot_h = sole_pos[:, 2]                                           # ground is at z = 0
        force_bw = self.foot_force() / self.dog_weight
        step_energy = self._step_energy.clone()
        self._step_energy.zero_()

        # swing reference: a smooth arc from 0 up to the commanded height and back to 0 at touchdown
        s = ((self.phase - self.cfg.duty_factor) / (1.0 - self.cfg.duty_factor)).clamp(0.0, 1.0)
        h_ref = self.swing_h * torch.sin(math.pi * s) * swing.float()
        gait_match = (contact == stance).float()
        slip = torch.norm(sole_vel[:, :2], dim=-1) * contact.float()
        vx = self.slider_vx()

        dt = self.step_dt
        dz = z - self.cfg.stand_z
        if self.cfg.height_mode == "floor":      # hopping: rising into a flight is fine, only sagging below costs
            dz = torch.clamp(dz, max=0.0)
        rewards = {
            "height": torch.exp(-torch.square(dz) / self.cfg.height_sigma**2) * self.cfg.rew_height * dt,
            "gait": gait_match * self.cfg.rew_gait * dt,
            "clearance": swing.float() * torch.exp(-torch.square(foot_h - h_ref) / self.cfg.clearance_sigma**2)
            * self.cfg.rew_clearance * dt,
            "energy": step_energy * self.cfg.rew_energy,
            "impact": torch.clamp(force_bw - self.cfg.impact_threshold_bw, min=0.0) * self.cfg.rew_impact * dt,
            "foot_slip": slip * self.cfg.rew_foot_slip * dt,
            "action_rate": torch.sum(torch.square(self.actions - self.prev_actions), dim=1)
            * self.cfg.rew_action_rate * dt,
            "collapse": self.reset_terminated.float() * self.cfg.rew_collapse,
            "track_vel": torch.exp(-torch.square(vx - self.fwd_cmd) / self.cfg.vel_sigma**2) * self.cfg.rew_track_vel * dt,
        }
        for k, v in rewards.items():
            if k not in self._episode_sums:
                self._episode_sums[k] = torch.zeros(self.num_envs, device=self.device)
            self._episode_sums[k] += v

        # ---- metrics ----
        self._height_err_sum += torch.abs(dz)
        self._gait_match_sum += gait_match
        swing_start = swing & self.prev_stance
        swing_end = stance & ~self.prev_stance
        self._clear_short_sum += torch.clamp(self.swing_h - self._swing_peak, min=0.0) * swing_end.float()
        self._swings += swing_end.float()
        self._swing_peak = torch.where(swing_start, foot_h, torch.maximum(self._swing_peak, foot_h))
        self._impact_peak = torch.maximum(self._impact_peak, force_bw)
        self._energy_sum += step_energy
        self._vel_err_sum += torch.abs(vx - self.fwd_cmd)
        self._distance_sum += torch.abs(vx) * dt
        self.prev_stance = stance.clone()

        # ---- advance the gait clock for the next step ----
        self.phase = (self.phase + self.step_dt * self.gait_freq) % 1.0

        self._debug_steps += 1
        if self._debug_steps % 250 == 0:
            print(f"[STATUS] on the clock {gait_match.mean().item() * 100:.0f}% | "
                  f"body height err {torch.abs(z - self.cfg.stand_z).mean().item() * 100:.1f} cm | "
                  f"power {(step_energy / dt).mean().item():.0f} W | peak force {force_bw.max().item():.1f} BW | "
                  f"speed {vx.mean().item():.2f} m/s (cmd {self.fwd_cmd.mean().item():.2f})")
        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    # ---------------------------------------------------------------- episode end
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        collapsed = self.slider_z() < self.cfg.stand_z - self.cfg.collapse_drop
        return collapsed, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # ---- log the finished episodes ----
        steps = self.episode_length_buf[env_ids].float().clamp(min=1.0)
        ep_time = steps * self.step_dt
        extras = {f"Episode_Reward/{k}": torch.mean(v[env_ids]) / self.max_episode_length_s
                  for k, v in self._episode_sums.items()}
        extras["Episode/body_height_error_cm"] = 100.0 * torch.mean(self._height_err_sum[env_ids] / steps)
        extras["Episode/contact_match"] = torch.mean(self._gait_match_sum[env_ids] / steps)
        extras["Episode/clearance_error_cm"] = (100.0 * self._clear_short_sum[env_ids].sum()
                                                / self._swings[env_ids].sum().clamp(min=1.0))
        extras["Episode/torque_rms_max_Nm"] = torch.sqrt(self._tau_sq_sum[env_ids].sum(dim=0) / ep_time.sum()).max()
        extras["Episode/impact_peak_bw"] = torch.mean(self._impact_peak[env_ids])
        extras["Episode/mean_power_W"] = self._energy_sum[env_ids].sum() / ep_time.sum()
        if self.rig["type"] != "vertical":
            extras["Episode/vel_error_ms"] = torch.mean(self._vel_err_sum[env_ids] / steps)
            extras["Episode/mean_speed_ms"] = torch.mean(self._distance_sum[env_ids] / ep_time)
            # cost of transport = battery energy / (moving weight x distance): the standard efficiency number
            lift_dist = (self.moving_weight[env_ids] * self._distance_sum[env_ids]).sum().clamp(min=1e-3)
            extras["Episode/cost_of_transport"] = self._energy_sum[env_ids].sum() / lift_dist
        extras["Episode_Termination/collapsed"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        self.extras["log"] = extras
        for buf in (*self._episode_sums.values(), self._height_err_sum, self._gait_match_sum, self._swing_peak,
                    self._clear_short_sum, self._swings, self._impact_peak, self._energy_sum, self._vel_err_sum,
                    self._distance_sum, self._tau_sq_sum, self._step_energy):
            buf[env_ids] = 0.0

        super()._reset_idx(env_ids)

        # ---- new commands, clock restarts in stance ----
        k = len(env_ids)
        self.gait_freq[env_ids] = torch.empty(k, device=self.device).uniform_(*self.cfg.gait_freq_range)
        self.swing_h[env_ids] = torch.empty(k, device=self.device).uniform_(*self.cfg.swing_height_range)
        self.fwd_cmd[env_ids] = torch.empty(k, device=self.device).uniform_(*self.cfg.fwd_speed_range)
        self.phase[env_ids] = 0.0
        self.prev_stance[env_ids] = True
        self.push_timer[env_ids] = torch.empty(k, device=self.device).uniform_(*self.cfg.push_interval_s)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

        # ---- robot back to the start (standing pose, carriage at its start height) ----
        root = self.robot.data.default_root_state[env_ids]
        root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos[env_ids],
                                            self.robot.data.default_joint_vel[env_ids], None, env_ids)
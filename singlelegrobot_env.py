# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .singlelegrobot_env_cfg import SinglelegrobotEnvCfg
from isaaclab.utils.math import quat_apply

class SinglelegrobotEnv(DirectRLEnv):
    cfg: SinglelegrobotEnvCfg

    def __init__(self, cfg: SinglelegrobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # joints: leg motors (policy) and the slider (passive)
        self.dof_idx, dof_names = self.robot.find_joints(self.cfg.dof_names)
        self.slider_idx = self.robot.find_joints(["pz_.*"])[0][0]
        # the foot link, for the contact sensor
        self.foot_id = self._contact.find_bodies(self.cfg.foot_body)[0][0]
        self.foot_body_idx = self.robot.find_bodies(self.cfg.foot_body)[0][0]
        self.hip_body_idx = self.robot.find_bodies("hip_link")[0][0]
        self.sole_offset = torch.tensor(self.cfg.sole_offset, device=self.device).repeat(self.num_envs, 1)
        n = self.num_envs
        self.actions = torch.zeros(n, len(self.dof_idx), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        # hop task state
        self.target_height = torch.zeros(n, device=self.device)        # commanded hop height [m]
        self.in_air = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.liftoff_z = torch.zeros(n, device=self.device)            # carriage height at takeoff
        self.prev_vz = torch.zeros(n, device=self.device)
        self.last_rise = torch.zeros(n, device=self.device)            # height of the latest hop
        self.phase = torch.zeros(n, device=self.device)                # hop rhythm, 0..1 per cycle
        self._debug_steps = 0
        # per-episode reward sums -> TensorBoard "Episode_Reward/<name>"
        self._episode_sums = {k: torch.zeros(n, device=self.device)
                              for k in ("apex", "gait", "lift", "action_rate", "torque", "collapse","foot_slip", "foot_under_hip","joint_acc")}
        self._hops = torch.zeros(n, device=self.device)
        self.peak_height = torch.zeros(n,device=self.device) # highest point of the current hop ( torch.zeros creates a new tensor every time, so we can't use last_rise for this)
        self.air_steps = torch.zeros(n, device=self.device) # number of steps while the foot is in the air
        print(f"[INFO] Leg joints controlled by the policy: {dof_names}")
        print(f"[INFO] Links: {self.robot.body_names}")
        print(f"[INFO] Foot contact measured on: {self.cfg.foot_body}")

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact = ContactSensor(self.cfg.contact_sensor)
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, friction_combine_mode="max", restitution=0.0, restitution_combine_mode="min")),
        )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation and sensor to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---------------------------------------------------------------- readings
    def slider_z(self) -> torch.Tensor:
        """Carriage height relative to its start height [m]."""
        return self.robot.data.joint_pos[:, self.slider_idx]

    def slider_vz(self) -> torch.Tensor:
        """Carriage vertical speed [m/s]."""
        return self.robot.data.joint_vel[:, self.slider_idx]

    def foot_contact(self) -> torch.Tensor:
        """True where the foot is pressing on the ground."""
        force = self._contact.data.net_forces_w[:, self.foot_id].norm(dim=-1)
        return force > self.cfg.contact_threshold
    
    def sole_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World position and velocity of the bottom of the foot."""
        d = self.robot.data
        q = d.body_link_quat_w[:, self.foot_body_idx]
        r = quat_apply(q, self.sole_offset)     
        down = torch.zeros_like(r)
        down[:, 2] = -self.cfg.foot_radius                          # frame origin -> sole
        pos = d.body_link_pos_w[:, self.foot_body_idx] + r
        vel = d.body_link_lin_vel_w[:, self.foot_body_idx] + torch.cross(
            d.body_link_ang_vel_w[:, self.foot_body_idx], r, dim=-1)
        return pos, vel

    # ---------------------------------------------------------------- actions
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        default_pos = self.robot.data.default_joint_pos[:, self.dof_idx]
        targets = default_pos + self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target(targets, joint_ids=self.dof_idx)

    # ---------------------------------------------------------------- observations
    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos[:, self.dof_idx] - self.robot.data.default_joint_pos[:, self.dof_idx]
        joint_vel = self.robot.data.joint_vel[:, self.dof_idx]
        obs = torch.cat(
            (
                joint_pos,                                              # leg angles (relative to default)
                joint_vel,                                              # leg speeds
                self.slider_z().unsqueeze(-1),                          # carriage height
                self.slider_vz().unsqueeze(-1),                         # carriage vertical speed
                self.target_height.unsqueeze(-1),                       # the command: how high to hop
                torch.sin(2 * torch.pi * self.phase).unsqueeze(-1),     # where we are in the hop rhythm
                torch.cos(2 * torch.pi * self.phase).unsqueeze(-1),
                self.foot_contact().float().unsqueeze(-1),              # foot on the ground (1) / in the air (0)
                self.peak_height.unsqueeze(-1),                         # highest point of the current hop
            ),
            dim=-1,
        )
        return {"policy": obs}

    # ---------------------------------------------------------------- hop measurement
    def _update_hops(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (apex, rise): apex = True where a hop peaked this step, rise = that hop's height [m]."""
        z, contact = self.slider_z(), self.foot_contact()
        airborne = contact.logical_not()
        took_off = airborne & self.in_air.logical_not()  # was on the ground, now in the air
        landed = contact & self.in_air                      # was in the air, now on the ground
        self.peak_height = torch.where(took_off, z, self.peak_height)  # reset the peak height at liftoff
        self.air_steps = torch.where(airborne, self.air_steps +1 , self.air_steps) # count the number of steps in the air
        long_enough = self.air_steps * self.step_dt > self.cfg.min_air_time # only count a hop if the foot was in the air for at least this long
        hop = landed & long_enough
        rise = torch.where(hop, self.peak_height - self.liftoff_z, torch.zeros_like(z))
        self.last_rise = torch.where(hop, rise, self.last_rise)
        # update the hop state for the next step
        self.air_steps = torch.where(contact, torch.zeros_like(self.air_steps), self.air_steps)
        self.liftoff_z = torch.where(contact, z, self.liftoff_z)
        self.in_air = airborne
        return hop, rise

    # ---------------------------------------------------------------- rewards
    def _get_rewards(self) -> torch.Tensor:
        hop, rise = self._update_hops()

        # 1) reach the target height: paid once per hop, at its top
        err = rise - self.target_height
        hop_score = torch.exp(-torch.square(err) / self.cfg.apex_sigma**2)

        # 2) hop rhythm: foot on the ground for the first part of each cycle, in the air for the rest
        flight_time = 2.0 * torch.sqrt(2.0 * self.target_height / 9.81)
        period = self.cfg.stance_time + flight_time
        self.phase = (self.phase + self.step_dt * self.cfg.hop_frequency) % 1.0
        want_air = self.phase >= self.cfg.stance_fraction
        gait_match = (self.in_air == want_air).float() # 1.0 if the foot is in the right state (on ground / in air) for this part of the hop cycle, else 0.0
        sole_pos, sole_vel = self.sole_state() # sole_pos = world position of the bottom of the foot, sole_vel = world velocity of the bottom of the foot
        contact = self.foot_contact().float() # contact detection is noisy, so we use the smoothed version for the gait reward
        slip = torch.norm(sole_vel[:, :2], dim=-1) * contact                    # sideways speed on the ground
        hip_x = self.robot.data.body_link_pos_w[:, self.hip_body_idx, 0]
        off_x = sole_pos[:, 0] - hip_x  
        rewards = {
            "apex": hop.float() * hop_score * self.cfg.rew_apex,
            "gait": gait_match * self.cfg.rew_gait * self.step_dt,
            "lift": torch.clamp(self.slider_vz(), min=0.0) * self.cfg.rew_lift * self.step_dt,
            "action_rate": torch.sum(torch.square(self.actions - self.prev_actions), dim=1)
            * self.cfg.rew_action_rate * self.step_dt,
            "torque": torch.sum(torch.square(self.robot.data.applied_torque[:, self.dof_idx]), dim=1)
            * self.cfg.rew_torque * self.step_dt,
            "collapse": self.reset_terminated.float() * self.cfg.rew_collapse,
            "foot_slip": slip * self.cfg.rew_foot_slip * self.step_dt,
            "foot_under_hip": (off_x < 0).float() * self.cfg.rew_foot_under_hip * self.step_dt,
            "joint_acc": torch.sum(torch.square(self.robot.data.joint_acc[:, self.dof_idx]), dim=1) 
            * self.cfg.rew_joint_acc * self.step_dt,    
        }
        for k, v in rewards.items():
            self._episode_sums[k] += v
        self._hops += hop.float()

        # status line every 250 steps (5 s): is contact detected, is it hopping?
        self._debug_steps += 1
        if self._debug_steps % 250 == 0:
            print(f"[STATUS] foot on ground {self.foot_contact().float().mean().item() * 100:.0f}% | "
                  f"rhythm match {gait_match.mean().item() * 100:.0f}% | "
                  f"carriage z {self.slider_z().mean().item():+.3f} m | hops this step {int(hop.sum().item())}")

        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    # ---------------------------------------------------------------- episode end
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        collapsed = self.slider_z() < self.cfg.collapse_height
        return collapsed, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # log the finished episodes for TensorBoard
        extras = {f"Episode_Reward/{k}": torch.mean(v[env_ids]) / self.max_episode_length_s
                  for k, v in self._episode_sums.items()}
        extras["Episode/hops_per_episode"] = torch.mean(self._hops[env_ids])
        extras["Episode/last_hop_error_cm"] = torch.mean(
            torch.abs(self.last_rise[env_ids] - self.target_height[env_ids])) * 100
        extras["Episode_Termination/collapsed"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        self.extras["log"] = extras
        for v in self._episode_sums.values():
            v[env_ids] = 0.0
        self._hops[env_ids] = 0.0

        super()._reset_idx(env_ids)

        # new target height for each reset robot, and clear the hop state
        lo, hi = self.cfg.target_height_range
        self.target_height[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(lo, hi)
        self.in_air[env_ids] = False
        self.liftoff_z[env_ids] = 0.0
        self.prev_vz[env_ids] = 0.0
        self.last_rise[env_ids] = 0.0
        self.phase[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0 # reset the previous actions to zero for each reset robot
        self.peak_height[env_ids] = 0.0 # reset the highest point of the current hop for each reset robot
        self.air_steps[env_ids] = 0.0 # reset the number of steps in the air for each reset robot

        # put the robot back at the start
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
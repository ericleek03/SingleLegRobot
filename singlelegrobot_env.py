# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .singlelegrobot_env_cfg import SinglelegrobotEnvCfg


class SinglelegrobotEnv(DirectRLEnv):
    cfg: SinglelegrobotEnvCfg

    def __init__(self, cfg: SinglelegrobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, dof_names = self.robot.find_joints(self.cfg.dof_names)
        self.slider_idx = self.robot.find_joints(["pz_.*"])[0][0]
        #self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)
        print(f"[INFO] Leg joints controlled by the policy: {dof_names}")
        print(f"[INFO] Leg Links controlled by the policy {self.robot.body_names}")

        

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        default_pos = self.robot.data.default_joint_pos[:, self.dof_idx] 
        targets = default_pos + self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target(targets, joint_ids = self.dof_idx)

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos[:, self.dof_idx] - self.robot.data.default_joint_pos[:, self.dof_idx]
        joint_vel = self.robot.data.joint_vel[:, self.dof_idx]
        slider_pos = self.robot.data.joint_pos[:, self.slider_idx].unsqueeze(dim=-1)
        slider_vel = self.robot.data.joint_vel[:, self.slider_idx].unsqueeze(dim=-1)
        observations = {"policy": torch.cat((joint_pos, joint_vel, slider_pos, slider_vel), dim=-1)}
        return observations

    def _update_hops(self):
        z, vz, contact = self.slider_z(), self.slider_vz(), self.foot_contact()
        self.liftoff_z = torch.where(contact, z, self.liftoff_z)
        self.in_air = contact.logical_not()
        apex = self.in_air & (self.prev_vz > 0.0) & (vz<=0.0)
        rise = torch.where(apex, z - self.liftoff_z, None)

    def _get_rewards(self) -> torch.Tensor:
        joint_pos_error = self.robot.data.joint_pos[:, self.dof_idx] - self.robot.data.default_joint_pos[:,self.dof_idx]
        total_reward = -torch.sum(torch.square(joint_pos_error), dim=-1)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return False, time_out
    """self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out"""

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward
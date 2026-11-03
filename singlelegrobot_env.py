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
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .singlelegrobot_env_cfg import SinglelegrobotEnvCfg, LEG_MASS
from isaaclab.utils.math import quat_apply


# creates torch tensors on the same device as the robot's data
class SinglelegrobotEnv(DirectRLEnv):
    cfg: SinglelegrobotEnvCfg

    def __init__(self, cfg: SinglelegrobotEnvCfg, render_mode: str | None = None, **kwargs):
        share = cfg.body_mass * cfg.load_share
        lo, hi = cfg.payload_range
        cfg.events.carriage_mass.params["mass_distribution_params"] = (lo * share - LEG_MASS, hi * share - LEG_MASS)
        super().__init__(cfg, render_mode, **kwargs)
        # joints: leg motors (policy) and the slider (passive)
        self.dof_idx, dof_names = self.robot.find_joints(self.cfg.dof_names)
        self.z_idx = self.robot.find_joints(["pz_z"])[0][0]
        try:
            self.x_idx = self.robot.find_joints("pz_x")[0][0] # planar rig
        except ValueError:
            self.x_idx = None # 2D rig

        self.foot_id = self._contact.find_bodies(self.cfg.foot_body)[0][0]
        self.foot_body_idx = self.robot.find_bodies(self.cfg.foot_body)[0][0]
        self.mount_body_idx = self.robot.find_bodies("mount_link")[0][0]
        base_idx = self.robot.find_bodies("base_link")[0][0]
        masses = self.robot.root_physx_view.get_masses().to(self.device) # num_envs, num_bodies
        self.moving_weight = (masses.sum(dim=1) - masses[:, base_idx]) * 9.81 # weight of the moving parts of the robot [N]
        self.dog_weight = self.cfg.body_mass * 9.81 # weight of the body (without the leg) [N]

        n = self.num_envs
        self.sole_offfset = torch.tensor(self.cfg.sole_offset, device=self.device).repeat(n,1)
        self.actions = torch.zeros(n, len(self.dof_idx), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

        # gait clock commands
        self.phase = torch.zeros(n, device=self.device) # 0.0 = start of stance phase, 1.0 = end of stance phase
        self.gait_frequency = torch.zeros(n, device=self.device) # [Hz] how many hops per second
        self.swing_height = torch.zeros(n, device=self.device) # [m] height of the swing leg above the ground
        self.fwd_cmd = torch.zeros(n, device=self.device) # [m/s] forward speed command
        self.prev_stance = torch.zeros(n, device = self.device)
        self.push_timer = torch.zeros(n, device=self.device)

        # energy / torque bookkeeping
        self._step_energy = torch.zeros(n, device=self.device)
        self._energy_sum = torch.zeros(n, device=self.device)
        self._tau_sq_sum = torch.zeros(n, len(self.dof_idx), device=self.device)

        #metric for tensorboard optuna

        self._height_err_sum = torch.zeros(n, device=self.device) # sum of the absolute error between the target height and the actual height at the apex of each hop
        self._gait_match_sum = torch.zeros(n, device=self.device) # sum of the gait match metric at each step
        self._swing_peak = torch.zeros(n, device=self.device) # highest point of the swing leg during each hop
        self._clear_short_sum = torch.zeros(n, device=self.device) # sum of the time the foot is in the air for each hop
        self._swings = torch.zeros(n, device=self.device) # number of swings completed
        self._impact_peak = torch.zeros(n, device=self.device) # highest impact force measured at the foot during each hop
        self._vel_err_sum = torch.zeros(n, device=self.device) # sum of the absolute error between the target velocity and the actual velocity at the apex of each hop
        self._episode_sums: dict[str, torch.Tensor] = {}
        self._debug_steps = 0

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
                static_friction=1.0, dynamic_friction=1.0, friction_combine_mode="multiply", restitution=0.0, restitution_combine_mode="min")),
        ) # spawn a ground plane with high friction
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation and sensor to scene
        self.scene.articulations["robot"] = self.robot # add the robot to the scene
        self.scene.sensors["contact_sensor"] = self._contact # add the contact sensor to the scene
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)) # dome light for the scene
        light_cfg.func("/World/Light", light_cfg) # add a dome light to the scene

    # ---------------------------------------------------------------- readings
    def slider_z(self) -> torch.Tensor:
        """Carriage height relative to its start height [m]."""
        return self.robot.data.joint_pos[:, self.slider_idx] # vertical position of the slider joint

    def slider_vz(self) -> torch.Tensor:
        """Carriage vertical speed [m/s]."""
        return self.robot.data.joint_vel[:, self.slider_idx] # vertical speed of the slider joint

    def foot_force(self) -> torch.Tensor:
        return self._contact.data.net_forces_w[:, self.foot_id].norm(dim=-1) # total force on the foot in world frame [N]
    
    def foot_contact(self) -> torch.Tensor:
        """True where the foot is pressing on the ground."""
        return self.foot_force() > self.cfg.contact_threshold # True where the foot is pressing on the ground

    def stance_phase(self) -> torch.Tensor:
        """True where the foot is on the ground, False where it is in the air."""
        return self.phase < self.cfg.stance_fraction # True where the foot is on the ground, False where it is in the air
    
    def sole_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World position and velocity of the bottom of the foot."""
        d = self.robot.data
        q = d.body_link_quat_w[:, self.foot_body_idx] # quaternion of the foot link frame
        r = quat_apply(q, self.sole_offset) # vector from the foot link frame to the sole in world frame
        r[:, 2] -= self.cfg.foot_radius                          # frame origin -> sole
        pos = d.body_link_pos_w[:, self.foot_body_idx] + r # position of the foot link frame + r
        vel = d.body_link_lin_vel_w[:, self.foot_body_idx] + torch.cross(
            d.body_link_ang_vel_w[:, self.foot_body_idx], r, dim=-1) # velocity of the foot link frame + angular velocity cross r
        return pos, vel

    # ---------------------------------------------------------------- actions
    def _pre_physics_step(self, actions: torch.Tensor) -> None: # happens before the physics step, but after the environment step
        self.prev_actions = self.actions.clone() # store the previous actions for the action rate reward
        self.actions = actions.clone() #  store the current actions for the next step

    def _apply_action(self) -> None: # is called decimation number of times per environment step 
        targets = self.robot.data.default_joint_pos[:, self.dof_idx] + self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target(targets,joint_ids = self.dof_idx)
        self._apply_support()
        tau = self.robot.data.applied_torque[:,self.dof_idx]
        self._step_energy += self.motor_power() * self.physics_dt # energy used by the motors this step [J]
        self._tausq_sum += tau**2 * self.physics_dt

    def _apply_support(self):
        """Virtual spring-damper support for the body by the leg, if support_mode="active"."""
        mode = self.cfg.support_mode
        if mode == "none":
            return
        elif mode == "active":
            # apply a spring-damper force to the body along the vertical axis of the leg
            swing = (-self.stance_phase()).float()
            f= (self.moving_weight + self.cfg.support_stiffness * (self.cfg.stand_z - self.slider_z()) - self.cfg.support_damping * self.slider_vz()) * swing
            - self.cfg.support_damping * self.slider_vz()
            f = torch.clamp(f, min=0.0) * swing # only push up, not down
        elif mode == "passive":
            f = self.cfg.passive_fraction * self.moving_weight
        else:
            raise ValueError(f"Unknown support_mode: {mode}")

        forces = torch.zeros(self.num_envs, 1,3, device=self.device)
        forces[:, 0, 2] = f
        self.robot.permanent_wrench_composer.set_forces_and_torques(forces = forces, body_ids = [self.mount_body_idx], is_global=True) 

    def _push_carriage(self): 
        self.push_timer -= self.step_dt
        ids = (self.push_timer <= 0.0).nonzero(as_tuple=False).flatten()
        if len(ids) == 0:
            return
        pos = self.robot.data.joint_pos[ids].clone()
        vel = self.robot.data.joint_vel[ids].clone()
        vel[:, self.z_idx] += torch.empty(len(ids), device=self.device).uniform_(-self.cfg.push_vel, self.cfg.push_vel)
        self.robot.write_joint_state_to_sim(pos, vel, env_ids = ids)
        lo, hi = self.cfg.push_interval_s
        self.push_timer[ids] = torch.empty(len(ids), device=self.device).uniform_(lo,hi)  
    # ---------------------------------------------------------------- observations
    def _get_observations(self) -> dict:
        d = self.robot.data
        # where i get my hyperparameters
        obs = torch.cat(
            (
                d.joint_pos[:, self.dof_idx] - d.default_joint_pos[:,self.dof_idx], # 2 leg angles vs standing
                d.joint_vel[:, self.dof_idx], # 2 leg speeds
                self.actions,
                torch.sin(2 * math.pi * self.phase).unsqueeze(-1),
                torch.cos(2 * math.pi * self.phase).unsqueeze(-1),
                self.gait_frequency.unsqueeze(-1),
                self.foot_contact().float().unsqueeze(-1),
                (self.slider_z() - self.cfg.stand_z).unsqueeze(-1),
                self.slider_vz().unsqueeze(-1),
                self.fwd_cmd.unsqueeze(-1),
                self.slider_vx().unsqueeze(-1),
                
                
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
        rewards = {
            "height": torch.exp(-torch.square(z - self.cfg.stand_z) / self.cfg.height_sigma**2) * self.cfg.rew_height * dt,
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
        self._height_err_sum += torch.abs(z - self.cfg.stand_z)
        self._gait_match_sum += gait_match
        swing_start = swing & self.prev_stance
        swing_end = stance & ~self.prev_stance
        self._clear_short_sum += torch.clamp(self.swing_h - self._swing_peak, min=0.0) * swing_end.float()
        self._swings += swing_end.float()
        self._swing_peak = torch.where(swing_start, foot_h, torch.maximum(self._swing_peak, foot_h))
        self._impact_peak = torch.maximum(self._impact_peak, force_bw)
        self._energy_sum += step_energy
        self._vel_err_sum += torch.abs(vx - self.fwd_cmd)
        self.prev_stance = stance.clone()
 
        # ---- advance the gait clock for the next step ----
        self.phase = (self.phase + self.step_dt * self.gait_freq) % 1.0
 
        self._debug_steps += 1
        if self._debug_steps % 250 == 0:
            print(f"[STATUS] on the clock {gait_match.mean().item() * 100:.0f}% | "
                  f"body height err {torch.abs(z - self.cfg.stand_z).mean().item() * 100:.1f} cm | "
                  f"power {(step_energy / dt).mean().item():.0f} W | peak force {force_bw.max().item():.1f} BW")
        return torch.sum(torch.stack(list(rewards.values())), dim=0)
    
   


    # ---------------------------------------------------------------- episode end
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        collapsed = self.slider_z() < self.cfg.collapse_height - self.cfg.collapse_drop
        return collapsed, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # log finished episodes
        steps = self.episode_length_buf[env_ids].float().clamp(min=1.0)
        ep_time = steps*self.step_dt
        extras = {f"Episode_Reward/{k}": torch.mean(v[env_ids]) / self.max_episode_length_s}
        extras["Episode/body_height_error_cm"] = 100.0 * torch.mean(self._height_err_sum[env_ids] / steps)
        extras["Episode/contact_match"] = torch.mean(self._gait_match_sum[env_ids] / steps)
        extras["Episode/clearance_error_cm"] = (100.0 * self._clear_short_sum[env_ids].sum()
                                                / self._swings[env_ids].sum().clamp(min=1.0))
        extras["Episode/torque_rms_max_Nm"] = torch.sqrt(self._tau_sq_sum[env_ids].sum(dim=0) / ep_time.sum()).max()
        extras["Episode/impact_peak_bw"] = torch.mean(self._impact_peak[env_ids])
        extras["Episode/mean_power_W"] = self._energy_sum[env_ids].sum() / ep_time.sum()
        if self.x_idx is not None:
            extras["Episode/vel_error_ms"] = torch.mean(self._vel_err_sum[env_ids] / steps)
        extras["Episode_Termination/collapsed"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        self.extras["log"] = extras
        for buf in (*self._episode_sums.values(), self._height_err_sum, self._gait_match_sum, self._swing_peak,
                    self._clear_short_sum, self._swings, self._impact_peak, self._energy_sum, self._vel_err_sum,
                    self._tau_sq_sum, self._step_energy):
            buf[env_ids] = 0.0
 
        super()._reset_idx(env_ids)

        # New comands
        k = len(env_ids)
        self.gait_frequency[env_ids] = torch.empty(k, device=self.device).uniform_(*self.cfg.gait_freq_range)
        self.swing_height[env_ids] = torch.empty(k, device = self.device).uniform_(*self.cfg.swing_height_range)
        self.fwd_cmd[env_ids] = torch.empty(k, device=self.device).uniform_(*self.cfg.fwd_speed_range)
        self.phase[env_ids] = 0.0
        self.prev_stance[env_ids] = True
        self.push_timer[env_ids] = torch.empty(k,device=self.device).uniform_(*self.cfg.push_interval_s)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

        # Robot back to standing position
        root = self.robot.data.default_root_state[env_ids]
        root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_top_sim(root[:,:7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:,:7], env_ids)
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos[env_ids],
                                            self.robot.data.default_joint_vel[env_ids], None, env_ids)
        
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sim import PhysxCfg, SimulationCfg
from SingleLegRobot.robots.single_leg import NUM_MOVING_JOINTS, SINGLE_LEG_CONFIG
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg


@configclass
class SinglelegrobotEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 10.0
    # - spaces definition
    action_space = NUM_MOVING_JOINTS
    observation_space = 2*NUM_MOVING_JOINTS + 6  # joint positions, joint velocities, base position
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation, physx=PhysxCfg(enable_external_forces_every_iteration=True),)

    # robot(s)
    robot_cfg: ArticulationCfg = SINGLE_LEG_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=2.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    dof_names = ["(?!pz_).*"]
    # - action scale
    action_scale = 0.75 # [N]

    #foot contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(prim_path = "/World/envs/env_.*/Robot/.*", update_period =0.0, history_length=3, track_air_time =True)

    foot_body = "knee_link"
    target_height_range = (0.05, 0.15) # commanded target height
    contact_threshold = 0.01 # [m] distance threshold to consider the foot in contact with the ground
    collapse_height = -0.36 # carriage below

    # Reward weights
    rew_apx = 10.0 # weight for apex height reward
    apex_sigma = 0.03
    rew_air_time = 2.0
    rew_action_rate = -0.005
    rew_torque = -1.0e-5
    rew_collapse = -5.0
    hop_frequency = 2.0 # [Hz] desired hopping frequency for reward shaping
    stance_fraction = 0.5 # desired stance fraction for reward shaping
    rew_gait = 3.0
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]
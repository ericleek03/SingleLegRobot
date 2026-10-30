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
from isaaclab.envs import mdp 
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
@configclass
class EventCfg:
    armature = EventTerm(
    func=mdp.randomize_joint_parameters, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names = ["hip_joint", "knee_joint"]),
        "armature_distribution_params": (0.5, 1.7), "operation": "scale"},
    )


@configclass
class SinglelegrobotEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 10.0
    # - spaces definition
    action_space = NUM_MOVING_JOINTS
    observation_space = 3*NUM_MOVING_JOINTS + 6  # joint positions, joint velocities, base position
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

       # ---- hop task ----
    foot_body = "foot_link"                     # the link that touches the ground
    target_height_range = (0.05, 0.15)          # commanded hop height, liftoff to apex [m]
    contact_threshold = 1.0                     # foot force above this [N] = on the ground
    collapse_height = -0.20                    # carriage below this = collapsed
    hop_frequency = 2.0                         # hops per second the rhythm asks for
    stance_time = 0.5                       # share of each cycle with the foot on the ground
    sole_offset = (0.0060, -0.0810, 0.3686)
    foot_radius = 0.025
    # ---- reward weights ----
    rew_apex = 10.0                             # per hop, scaled by how close it was to the target
    apex_sigma = 0.03                           # [m] a hop 3 cm off the target scores ~37 %
    rew_gait = 2.0                              # per second: foot on ground / in air on the rhythm
    rew_lift = 0.0                              # reward moving the carriage upward
    rew_action_rate = -0.02                   # smooth commands
    rew_torque = -1.0e-5                        # low motor effort
    rew_collapse = -5.0                         # once, when the leg collapses
    hop_frequency = 2.0 # [Hz] desired hopping frequency for reward shaping
    stance_fraction = 0.5 # desired stance time for reward shaping
    rew_foot_slip = -20.0
    rew_foot_under_hip = -300.0
    rew_joint_acc = -5.0e-7   # penalises fast jerks of joints
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]
    min_air_time = 0.1 # seconds

    events: EventCfg = EventCfg()


# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sim import PhysxCfg, SimulationCfg
from SingleLegRobot.robots.single_leg import NUM_MOVING_JOINTS, SINGLE_LEG_CONFIG
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg

LEG_MASS = 1.78 # [kg] mass of the leg (without the slider)

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
    decimation = 4 # number of simulation steps per environment step
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


    #foot contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(prim_path = "/World/envs/env_.*/Robot/.*", update_period =0.0, history_length=3, track_air_time =True)

    events: EventCfg = EventCfg()
     # custom parameters/scales
    # - controllable joint
    dof_names = ["(?!pz_).*"] # leg joints only
    action_scale = 0.50 # [rad] joint target=  standing pose + action_scale * action
    foot_body = "foot_link" # name of the foot link in the URDF
    sole_offset = (0.0060, -0.0810, 0.3686) # foot sphere centre in foot link frame [m]
    foot_radius = 0.025 # [m] radius of the foot sphere for contact detection
    contact_threshold = 1.0 # [N] threshold for foot contact detection

    body_mass = 12.0 # [kg] mass of the body (without the leg)
    load_share = 0.5 # fraction of the total mass supported by the leg (the rest is supported by the slider)
    payload_range = (0.8, 1.2) # carriage mass scaling range
    stand_z = -0.05 # [m] height of the foot above the ground when standing
    collapse_drop = 0.10 # [m] height of the foot above the ground when the leg is collapsed

    support_mode = "active" # "active" or "passive" support of the body by the leg
    passive_fraction = 0.5 # fraction of the body mass supported by the leg when support_mode="passive"
    support_stiffness = 1000.0 # [N/m] spring stiffness for passive support
    support_damping = 60.0 # [N*s/m] spring damping for passive support

    # commands
    gait_frquency_range = (1.5, 3.0) # [Hz] range of gait frequencies for the leg
    swing_height_range = (0.03, 0.07) # [m] range of swing heights for the leg
    duty_factor = 0.55 # shared fraction of the gait cycle spent in stance phase (0.5 = symmetric gait)

    fwd_speed_range = (0.0, 0.0) # [m/s] for forward speed range use for planar setup


    push_vel = 0.0 # [m/s] velocity of the push force applied to the robot
    push_interval = (1.0, 3.0) # [s] time interval between pushes

    # reward scales
    rew_height = 5.0 # exp(-(z-stand_z)^2 * rew_height) for foot height above ground
    height_sigma = 0.02 # [m] sigma for foot height reward
    rew_gait = 2.0 # foot down/up reward scale
    rew_clearance = 10 # foot follows a smooth trajectory reward scale
    clearance_sigma = 0.02 # [m] sigma for foot clearance reward
    rew_energy = -0.02 # energy penalty scale per J
    rew_impact = -0.1 # impact penalty scale per N
    impact_threshold = 10.0 # [N] threshold for impact penalty
    rew_foot_slip = -20.0 # foot slip penalty scale per m
    rew_action_rate = -0.01 # action rate penalty scale per (rad/s)^2
    rew_collapse = -3.0 # penalty for foot height below collapse_drop
    rew_track_vel = 0.00 #for planar setup, reward for tracking the commanded velocity
    vel_sigma = 0.25 # [m/s] sigma for velocity tracking reward

    # motor energy model GIM8108
    joule_coeff = 0.5 # [J/(N*m)^2] Joule heating coefficient for the motors
    regen = False # True to enable regenerative braking, False to disable it


# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One leg of a trotting dog, on the vertical slider (planar-ready).

The carriage carries this leg's share of the dog. A gait clock sets stance / swing. During swing, a virtual
"other legs" force holds the carriage (support_mode="active"), like the diagonal pair does in a real trot.
Each episode commands a step frequency and swing height, so the leg learns the range the dog will use.
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from SingleLegRobot.robots.single_leg import SINGLE_LEG_CONFIG

LEG_MASS = 1.78     # base_link + hip_link + knee_link + foot_link [kg], from the URDF


@configclass
class EventCfg:
    """Robustness: every robot gets slightly different physics, so the policy can't overfit one exact world."""

    armature = EventTerm(
        func=mdp.randomize_joint_parameters, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_joint", "knee_joint"]),
                "armature_distribution_params": (0.5, 1.7), "operation": "scale"},
    )
    # carriage = this leg's share of the dog (minus the leg itself), +-payload_range. The env fills in the numbers
    # from body_mass / load_share / payload_range, so command-line overrides of those work.
    carriage_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["mount_link"]),
                "mass_distribution_params": (4.2, 4.2), "operation": "abs"},
    )
    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["foot_link"]),
                "static_friction_range": (0.6, 1.2), "dynamic_friction_range": (0.5, 1.0),
                "restitution_range": (0.0, 0.0), "num_buckets": 64, "make_consistent": True},
    )
    motor_gains = EventTerm(
        func=mdp.randomize_actuator_gains, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_joint", "knee_joint"]),
                "stiffness_distribution_params": (0.9, 1.1), "damping_distribution_params": (0.9, 1.1),
                "operation": "scale"},
    )


@configclass
class SinglelegrobotEnvCfg(DirectRLEnvCfg):
    # ---- env ----
    decimation = 4                     # 200 Hz physics, 50 Hz policy
    episode_length_s = 10.0
    action_space = 2                   # hip, knee
    observation_space = 15             # see _get_observations
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation,
                                       physx=PhysxCfg(enable_external_forces_every_iteration=True))
    robot_cfg: ArticulationCfg = SINGLE_LEG_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=2.0, replicate_physics=True)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", update_period=0.0, history_length=3, track_air_time=True
    )
    events: EventCfg = EventCfg()

    # ---- control / robot ----
    dof_names = ["(?!pz_).*"]          # leg joints only, never the slider
    action_scale = 0.7                 # [rad] joint target = standing pose + action_scale * action (action in +-1)
    foot_body = "foot_link"
    sole_offset = (0.0060, -0.0810, 0.3686)   # foot sphere centre in foot_link's frame [m]
    foot_radius = 0.025                # [m]
    contact_threshold = 1.0            # [N] foot force above this = on the ground

    # ---- the dog this leg belongs to ----
    body_mass = 12.0                   # [kg] planned dog mass
    load_share = 0.5                   # trot: 2 legs in stance -> each carries ~half the dog
    payload_range = (0.8, 1.2)         # carriage mass randomised to 80..120 % of the share
    stand_z = -0.06                   # [m] slider position when standing 
    collapse_drop = 0.10               # [m] carriage more than this below stand_z = collapsed

    # ---- "other legs" support (must match the REAL rig you build) ----
    support_mode = "active"            # "active": holds the carriage in swing | "passive": constant | "none"
    passive_fraction = 0.5             # passive only: share of the carriage weight carried all the time
    support_stiffness = 800.0         # [N/m]   active: spring about stand_z during swing
    support_damping = 60.0             # [N s/m]

    # ---- commands (sampled every episode) ----
    gait_freq_range = (1.5, 3.0)       # [Hz] steps per second
    swing_height_range = (0.03, 0.07)  # [m]  peak foot height in swing
    duty_factor = 0.55                 # share of each cycle in stance (trot ~0.5-0.6, slow walk ~0.75)
    fwd_speed_range = (0.0, 0.0)       # [m/s] planar rig only (pz_x joint), e.g. (0.0, 0.5); 0 on the slider

    # ---- disturbances ----
    push_vel = 0.0                     # [m/s] random kick to the carriage's vertical speed; 0 = off (try 0.3 later)
    push_interval_s = (3.0, 6.0)       # [s] time between kicks

    # ---- rewards (per second unless noted; rew_height is the fixed reference) ----
    rew_height = 3.0                   # exp(-(z - stand_z)^2 / height_sigma^2): carry the load
    height_sigma = 0.06                # [m]
    rew_gait = 4.0                     # foot down/up when the clock says stance/swing
    rew_clearance = 4.0                # foot follows a smooth swing arc up to the commanded height
    clearance_sigma = 0.05             # [m]
    rew_energy = -0.002                 # per J of battery energy (not per second)
    rew_impact = -0.005                  # touchdown force above impact_threshold_bw (in dog body weights)
    impact_threshold_bw = 2.5
    rew_foot_slip = -10.0              # per m/s of sole sliding while in contact
    rew_action_rate = -0.005            # smooth commands
    rew_collapse = -5.0                # once, when it collapses
    rew_track_vel = 0.0                # planar rig only: exp(-(vx - cmd)^2 / vel_sigma^2); keep 0 on the slider
    vel_sigma = 0.25                   # [m/s]

    # ---- motor energy model (GIM8108-6) ----
    joule_coeff = 0.5                  # [W/Nm^2] copper heat = joule_coeff * torque^2 (R/Kt^2 = 0.73/1.22^2)
    regen = False                      # True only if the drivers return braking energy to the battery
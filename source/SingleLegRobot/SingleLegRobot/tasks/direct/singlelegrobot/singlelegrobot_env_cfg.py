# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One leg of the dog HOPPING FORWARD around the RADIAL BOOM (2DOFSinglelegmodel_boom.urdf from
scripts/make_boom_urdf.py: rotating base + hinged arm + tip linkage + 6.8 kg counterweight).
Like the real boom, nothing holds the leg up between pushes (support_mode="none"): it must hop - push off, fly,
land - while travelling round the circle at the commanded speed. The counterweight is in the URDF.
The env detects the rig from the URDF, so the same files also run the rails URDF (pz_x + pz_z) and the vertical
slider (pz_z only; then set fwd_speed_range = (0, 0) and rew_track_vel = 0). For a supported trot instead:
support_mode="active", duty_factor=0.55, height_mode="level".

The carriage carries this leg's share of the dog. A gait clock sets stance / swing. During swing, a virtual
"other legs" force holds the carriage up and keeps it moving at the commanded speed (support_mode="active"),
like the diagonal pair does in a real trot. During stance this leg carries its share and pushes the body forward.
Each episode commands a speed, step frequency and swing height, so the leg learns the range the dog will use.

Rewards below are STAGE 1 (learn to trot). Stage 2 (efficiency), resume with:
    env.rew_energy=-0.02 env.rew_impact=-0.1 env.rew_foot_slip=-20.0 env.rew_action_rate=-0.02
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

LEG_MASS = 2.05     # base_link + hip_link + knee_link + foot_link [kg], from the URDF


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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=4.0, replicate_physics=True)
    # 4 m apart so each boom's circle (radius 1.5 m) stays in its own patch
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", update_period=0.0, history_length=3, track_air_time=True
    )
    events: EventCfg = EventCfg()

    # ---- control / robot ----
    dof_names = ["(?!pz_).*"]          # leg joints only, never the slider
    action_scale = 0.7                 # [rad] joint target = standing pose + action_scale * action (action in +-1)
                                       #       0.7 -> foot can lift ~8 cm (0.4 only reached 4.5 cm)
    foot_body = "foot_link"
    sole_offset = (0.0060, -0.0810, 0.3686)   # foot sphere centre in foot_link's frame [m]
    foot_radius = 0.025                # [m]
    contact_threshold = 1.0            # [N] foot force above this = on the ground

    # ---- the dog this leg belongs to ----
    body_mass = 12.0                   # [kg] planned dog mass
    load_share = 0.5                   # trot: 2 legs in stance -> each carries ~half the dog
    payload_range = (0.8, 1.2)         # carriage mass randomised to 80..120 %
    mount_mass = None                  # None: dog-load test (the leg carries load_share x body_mass)
                                       # a number [kg]: your real mount hardware, so the counterweight lightens it
    stand_z = -0.06                    # [m] slider position when standing on hip 0.80 / knee 0.85 with the planar
                                       #     URDF (pz_z starts at 0.42); check with zero_agent, the leg sags a little
    collapse_drop = 0.10               # [m] carriage more than this below stand_z = collapsed
    start_height = 0.42                # [m] leg mount height at the start (boom pivot height / pz_z origin)

    # ---- "other legs" support (must match the REAL rig you build) ----
    support_mode = "none"              # real boom: nothing holds the leg between pushes (counterweight is in the URDF)
                                       # "active": virtual other legs in swing (supported trot) | "passive": constant
    passive_fraction = 0.5             # passive only: share of the carriage weight carried all the time
    support_stiffness = 500.0          # [N/m]   active: spring about stand_z during swing (soft: body may bob ~1 cm)
    support_damping = 60.0             # [N s/m]
    support_damping_x = 100.0          # [N s/m] active + planar: keeps the body at the commanded speed during swing

    # ---- commands (sampled every episode) ----
    gait_freq_range = (2.8, 3.5)       # [Hz] hops per second (2 Hz -> 0.3 s flights, 3 Hz -> 0.2 s)
    swing_height_range = (0.03, 0.07)  # [m]  peak foot height in swing
    duty_factor = 0.4                  # share of each cycle on the ground: < 0.5 = hopping with a flight phase
                                       # (supported trot ~0.55)
    fwd_speed_range = (0.1, 0.4)       # [m/s] commanded forward speed (planar rig). Flip the sign if the leg looks
                                       #       "backwards-kneed" compared with how it sits on the dog

    # ---- disturbances ----
    push_vel = 0.0                     # [m/s] random kick to the carriage's vertical speed; 0 = off (try 0.3 later)
    push_interval_s = (3.0, 6.0)       # [s] time between kicks

    # ---- rewards (per second unless noted; rew_height is the fixed reference) ----
    rew_height = 2.0                   # exp(-dz^2 / height_sigma^2): carry the load
    height_mode = "floor"              # "floor": only sagging below stand_z costs (hopping) | "level": keep level (trot)
    height_sigma = 0.04                # [m] 2 cm off still scores 78 %
    rew_gait = 4.0                     # foot down/up when the clock says stance/swing
    rew_clearance = 4.0                # foot follows a smooth swing arc up to the commanded height
    clearance_sigma = 0.05             # [m] wide: any lift of the foot already pays
    rew_energy = -0.002                # per J of battery energy (not per second). Stage 2: -0.02
    rew_impact = -0.05                 # touchdown force above impact_threshold_bw (dog body weights). Stage 2: -0.1
    impact_threshold_bw = 2.5
    rew_foot_slip = -5.0               # per m/s of sole sliding while in contact. Stage 2: -20
    rew_action_rate = -0.005           # smooth commands. Stage 2: -0.02
    rew_collapse = -5.0                # once, when it collapses
    rew_track_vel = 3.0                # planar: exp(-(vx - cmd)^2 / vel_sigma^2). 0 on the vertical slider
    vel_sigma = 0.15                   # [m/s]

    # ---- motor energy model (GIM8108-6) ----
    joule_coeff = 0.5                  # [W/Nm^2] copper heat = joule_coeff * torque^2 (R/Kt^2 = 0.73/1.22^2)
    regen = False                      # True only if the drivers return braking energy to the battery
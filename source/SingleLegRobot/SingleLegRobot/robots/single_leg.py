import os
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

URDF_PATH = os.path.join(os.path.dirname(__file__), "data", "2DOFSinglelegmodel.urdf")
MOUNT_HEIGHT = 0.0                      # the rig's base sits on the ground [m]

STIFFNESS = 42.0                        # PD position gain [N*m/rad]
DAMPING = 3.5                           # PD velocity gain [N*m*s/rad]
EFFORT_LIMIT = 18.25                    # motor torque limit [Nm]
VELOCITY_LIMIT = 363 * (2 * 3.14159 / 60)   # motor velocity limit [rad/s]
ARMATURE = 0.006

if not os.path.isfile(URDF_PATH):
    raise FileNotFoundError(f"URDF file not found: {URDF_PATH}")
_JOINTS = {j.get("name"): j for j in ET.parse(URDF_PATH).getroot().findall("joint")}

# ---- standing pose ----
# The same leg pose needs a different hip angle depending on how the hip housing is mounted:
#   mount_joint rpy "3.14159265 0 0"          -> housing bracket up       -> hip 0.80
#   mount_joint rpy "0 -1.57079633 3.14159265" -> housing bracket backward -> hip -0.771
# (the housing turned 90 deg about the hip axis, so the hip angle shifts by 90 deg too)
_mount_rpy = [float(v) for v in _JOINTS["mount_joint"].find("origin").get("rpy").split()]
_bracket_back = abs(_mount_rpy[1] + 1.5708) < 0.01
DEFAULT_JOINT_POSITIONS = {"hip_joint": -0.771 if _bracket_back else 0.80, "knee_joint": 0.85}

# starting pose: the leg in its standing pose + (boom rig) the boom tilted down 0.04 rad so the foot starts on
# the ground instead of dropping 6 cm at every reset (1.5 m x sin 0.04 = 6 cm)
INIT_JOINT_POS = dict(DEFAULT_JOINT_POSITIONS)
if "pz_pitch" in _JOINTS:
    INIT_JOINT_POS.update({"pz_pitch": 0.04, "pz_level": -0.04})


def count_moving_joints() -> int:
    """Leg joints the policy controls (rig joints start with pz_ and are not counted)."""
    return sum(1 for name, j in _JOINTS.items()
               if j.get("type") in ("revolute", "prismatic", "continuous") and not name.startswith("pz_"))


NUM_MOVING_JOINTS = count_moving_joints()

ACTUATORS = {
    "leg_motors": ImplicitActuatorCfg(
        joint_names_expr=["(?!pz_).*"],         # every joint EXCEPT the rig (pz_...)
        stiffness=STIFFNESS,
        damping=DAMPING,
        armature=ARMATURE,
        effort_limit_sim=EFFORT_LIMIT,
        velocity_limit_sim=VELOCITY_LIMIT,
    ),
    "slider": ImplicitActuatorCfg(
        joint_names_expr=["pz_(?!level).*"],    # free rig joints (slider / boom yaw + pitch), NOT the leveler
        stiffness=0.0,                          # no motor: free to move
        damping=0.05,                           # a little friction
    ),
}
if "pz_level" in _JOINTS:
    # the boom's parallelogram linkage: the env sets its target to -pitch every step, so the leg stays upright.
    # It needs a stiff drive; with stiffness 0 the leg would swing freely around the end of the boom.
    ACTUATORS["boom_level"] = ImplicitActuatorCfg(
        joint_names_expr=["pz_level"],
        stiffness=5000.0,
        damping=60.0,
    )

SINGLE_LEG_CONFIG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        fix_base=True,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=STIFFNESS, damping=DAMPING)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, MOUNT_HEIGHT),
        joint_pos=INIT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators=ACTUATORS,
)
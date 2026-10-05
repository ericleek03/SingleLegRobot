import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import xml.etree.ElementTree as ET

URDF_PATH = os.path.join(os.path.dirname(__file__), "data", "urdf" "URDFSingleLegModel.urdf")
MOUNT_HEIGHT = 0.8 # height of the leg top link above ground [m]
DEFAULT_JOINT_POSITIONS = [0.0, 0.0, 0.0] # default joint positions for the leg [rad]

STIFFNESS = 25.0  # PD Positions gain [N*m/rad]
DAMPING = 0.5  # PD Velocities gain [N*m*s/rad]
EFFORT_LIMIT = 23.5 # motor torue limit [Nm]
VELOCITY_LIMIT = 10.0 # motor velocity limit [rad/s]


def count_moving_joints(urdf_path: str) -> int:
    """
    Count the number of moving joints in a URDF file.

    Args:
        urdf_path (str): Path to the URDF file."""
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    joints = ET.parse(urdf_path).getroot().findall('joint')

    return sum(1 for j in joints if j.get("type") in ["revolute", "prismatic", "continuous"])

NUM_MOVING_JOINTS = count_moving_joints(URDF_PATH)

SINGLE_LEG_CONFIG = ArticulationCfg(
    spawn=sim_utils.UrdfSpawnCfg(asset_path=URDF_PATH, fixed_base=True),
    merge_fixed_joints=True,
    activate_contact_sensors=True,
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainscfg(
            stiffness=STIFFNESS, damping=DAMPING
        )
    ),
    articulation_props=sim_utils.UrdfConveterCfg.ArticlationPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=1,
        max_depenetration_velocity=10.0,
        max_angular_velocity=100.0,
    ),
    init_state=ArticulationCfg.InitStateCfg(
        pos=(0.0, 0.0, MOUNT_HEIGHT), joint_pos=DEFAULT_JOINT_POSITIONS
    ),
                                                                                                              joint_limits = sim_utils.UrdfConverterCfg.JointLimitsCfg(effort_limit = EFFORT_LIMIT, velocity_limit = VELOCITY_LIMIT),  actuators = {"leg_motors": ImplicitActuatorCfg(joint_names_exr=[".*"], stiffness = STIFFNESS, damping = DAMPING, effort_limit = EFFORT_LIMIT, velocity_limit = VELOCITY_LIMIT, control_type = "effort")})                                    
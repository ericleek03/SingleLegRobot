import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import xml.etree.ElementTree as ET

URDF_PATH = os.path.join(os.path.dirname(__file__), "data", "2DOFSinglelegmodel.urdf")
MOUNT_HEIGHT = 0.0 # height of the leg top link above ground [m]
DEFAULT_JOINT_POSITIONS = {"hip_joint": -0.6, "knee_joint": 1.2}# default joint positions for the leg [rad]

STIFFNESS = 30.0  # PD Positions gain [N*m/rad]
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

    return sum(1 for j in joints if j.get("type") in ("revolute", "prismatic", "continuous") and not j.get("name").startswith("pz_"))

NUM_MOVING_JOINTS = count_moving_joints(URDF_PATH)

SINGLE_LEG_CONFIG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(asset_path=URDF_PATH, fix_base=True,
    merge_fixed_joints=False,
    activate_contact_sensors=True,
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=STIFFNESS, damping=DAMPING)
        ),
    
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, MOUNT_HEIGHT), joint_pos=DEFAULT_JOINT_POSITIONS,),

    actuators={
        "leg_motors": ImplicitActuatorCfg(
            joint_names_expr=["(?!pz_).*"],       # every joint EXCEPT ones starting with pz_
            stiffness=STIFFNESS,
            damping=DAMPING,
            effort_limit_sim=EFFORT_LIMIT,
            velocity_limit_sim=VELOCITY_LIMIT,
        ),
        "slider": ImplicitActuatorCfg(
            joint_names_expr=["pz_.*"],           # the slider
            stiffness=0.0,                        # no motor: free to slide
            damping=0.05,                         # a little friction
        ),
    },
)
# ltl/top.py
"""
Simple script to run VLFM episodes with easy configuration.
Uses the same infrastructure as vlfm/run.py but configured in Python.
"""

import os
import sys
import json
import gzip
from typing import Optional, List
import socket

# Add project root to path
PROJECT_ROOT = "/home/dylan/PycharmProjects/VLM_LTL/vlfm"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # Important for relative paths

# Import all registration modules (same as run.py)
import frontier_exploration  # noqa
import hydra  # noqa
from habitat import get_config  # noqa
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat_baselines.run import execute_exp
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import vlfm.measurements.traveled_stairs  # noqa: F401
import vlfm.obs_transformers.resize  # noqa: F401
import vlfm.policy.action_replay_policy  # noqa: F401
import vlfm.policy.habitat_policies  # noqa: F401
import vlfm.utils.vlfm_trainer  # noqa: F401

# ============================================================================
# CONFIGURATION - Change these parameters to run different episodes
# ============================================================================

SCENE_NAME = "4ok3usBNeis"  # Change scene here
EPISODE_INDEX = 0  # Change episode index here (0, 1, 2, ...)

# Override parameters (None = use original episode values)
TARGET_OBJECT_OVERRIDE: Optional[str] = "chair"  # e.g., "chair", "bed", "plant", "toilet", "tv", "sofa"
START_POSITION_OVERRIDE: Optional[List[float]] = None  # e.g., [0.5, 0.0, 3.9]
START_ROTATION_OVERRIDE: Optional[List[float]] = None  # e.g., [0, 0.23, 0, 0.97] (quaternion)

# Episode settings
MAX_STEPS = 500
SAVE_VIDEO = True
VIDEO_FPS = 10

# Output directory
OUTPUT_DIR = "/home/dylan/PycharmProjects/VLM_LTL/vlfm/ltl/outputs"

# ============================================================================
# DATA PATHS
# ============================================================================

DATA_DIR = f"{PROJECT_ROOT}/data"
DATASET_DIR = f"{DATA_DIR}/datasets/objectnav/hm3d/v1/val"
CONFIG_DIR = f"{PROJECT_ROOT}/config"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_episode_info(scene_name: str, episode_index: int) -> dict:
    """Load episode information from JSON file."""
    scene_file = f"{DATASET_DIR}/content/{scene_name}.json.gz"

    if not os.path.exists(scene_file):
        scene_file = f"{DATASET_DIR}/content/{scene_name}.json"

    if not os.path.exists(scene_file):
        raise FileNotFoundError(f"Scene file not found: {scene_file}")

    # Load episodes
    if scene_file.endswith('.gz'):
        with gzip.open(scene_file, 'rt') as f:
            data = json.load(f)
    else:
        with open(scene_file, 'r') as f:
            data = json.load(f)

    episodes = data['episodes']

    if episode_index >= len(episodes):
        raise IndexError(f"Episode index {episode_index} out of range. Scene has {len(episodes)} episodes.")

    return episodes[episode_index]


def print_episode_info(episode: dict):
    """Print detailed episode information."""
    print("\n" + "=" * 80)
    print("EPISODE INFORMATION")
    print("=" * 80)
    print(f"Episode ID: {episode.get('episode_id', 'N/A')}")
    print(f"Scene ID: {episode.get('scene_id', 'N/A')}")
    print(f"\nStart Position: {episode.get('start_position', 'N/A')}")
    print(f"Start Rotation: {episode.get('start_rotation', 'N/A')}")

    obj_cat = episode.get('object_category', 'N/A')
    print(f"\nTarget Object Category: {obj_cat}")

    if 'info' in episode:
        print(f"\nAdditional Info:")
        for key, value in episode['info'].items():
            print(f"  {key}: {value}")

    if 'goals' in episode:
        print(f"\nGoal Positions:")
        for i, goal in enumerate(episode['goals']):
            print(f"  Goal {i}: {goal.get('position', 'N/A')}")

    print("=" * 80 + "\n")


def check_vlm_servers():
    """Check if VLM servers are running."""
    ports = [12181, 12182, 12183, 12184]
    server_names = ["GroundingDINO", "BLIP2-ITM", "MobileSAM", "YOLOv7"]

    print("\nChecking VLM Servers:")
    print("-" * 50)

    all_running = True
    for port, name in zip(ports, server_names):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()

        if result == 0:
            print(f"✓ Port {port} ({name}): Running")
        else:
            print(f"✗ Port {port} ({name}): NOT running")
            all_running = False

    print("-" * 50)

    if not all_running:
        print("\n⚠ WARNING: Some VLM servers are not running!")
        print("Please run: ./scripts/launch_vlm_servers.sh")
        return False
    else:
        print("✓ All VLM servers are running!\n")
        return True


def build_config_overrides(episode: dict) -> list:
    """Build Hydra config overrides list (mimics bash command arguments)."""

    overrides = [
        # Basic settings
        "habitat_baselines.evaluate=True",
        "habitat_baselines.test_episode_count=1",
        "habitat_baselines.num_environments=1",
        "habitat_baselines.verbose=True",

        # Video settings
        f"habitat_baselines.video_dir={OUTPUT_DIR}",
        f"habitat_baselines.video_fps={VIDEO_FPS}",
        '+habitat_baselines.eval.video_option=["disk"]',

        # Dataset settings
        "habitat.dataset.split=val",
        f'habitat.dataset.content_scenes=["{SCENE_NAME}"]',

        # Environment settings
        f"habitat.environment.max_episode_steps={MAX_STEPS}",

        # Map visualization settings
        f"habitat.task.measurements.frontier_exploration_map.max_episode_steps={MAX_STEPS * 2}",
        "habitat.task.measurements.frontier_exploration_map.map_resolution=1024",
        "habitat.task.measurements.frontier_exploration_map.draw_source=True",
        "habitat.task.measurements.frontier_exploration_map.draw_border=True",
        "habitat.task.measurements.frontier_exploration_map.draw_shortest_path=True",
        "habitat.task.measurements.frontier_exploration_map.draw_waypoints=True",
        "habitat.task.measurements.frontier_exploration_map.draw_goal_positions=True",

        # Fog of war
        "habitat.task.measurements.frontier_exploration_map.fog_of_war.draw=True",
        "habitat.task.measurements.frontier_exploration_map.fog_of_war.visibility_dist=5.0",
        "habitat.task.measurements.frontier_exploration_map.fog_of_war.fov=90",
    ]

    return overrides


def run_episode():
    """Main function to run a single episode."""

    print("\n" + "=" * 80)
    print("VLFM SINGLE EPISODE RUNNER")
    print("=" * 80)

    # Check prerequisites
    if not os.path.isdir(DATA_DIR):
        print(f"❌ ERROR: Data directory not found: {DATA_DIR}")
        return

    if not os.path.isfile(f"{DATA_DIR}/dummy_policy.pth"):
        print("❌ ERROR: Dummy policy weights not found!")
        print("Run: python -m vlfm.utils.generate_dummy_policy")
        return

    # Check VLM servers
    if not check_vlm_servers():
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    # Load episode
    try:
        print(f"\nLoading episode {EPISODE_INDEX} from scene {SCENE_NAME}...")
        episode = load_episode_info(SCENE_NAME, EPISODE_INDEX)
        print_episode_info(episode)
    except Exception as e:
        print(f"❌ ERROR loading episode: {e}")
        import traceback
        traceback.print_exc()
        return

    # Apply overrides
    overrides_applied = []
    if TARGET_OBJECT_OVERRIDE is not None:
        episode['object_category'] = TARGET_OBJECT_OVERRIDE
        overrides_applied.append(f"Target Object: {TARGET_OBJECT_OVERRIDE}")

    if START_POSITION_OVERRIDE is not None:
        episode['start_position'] = START_POSITION_OVERRIDE
        overrides_applied.append(f"Start Position: {START_POSITION_OVERRIDE}")

    if START_ROTATION_OVERRIDE is not None:
        episode['start_rotation'] = START_ROTATION_OVERRIDE
        overrides_applied.append(f"Start Rotation: {START_ROTATION_OVERRIDE}")

    if overrides_applied:
        print("\n" + "=" * 80)
        print("OVERRIDES APPLIED:")
        print("=" * 80)
        for override in overrides_applied:
            print(f"  • {override}")
        print("=" * 80 + "\n")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    # Initialize Hydra with absolute config directory path
    print("Initializing Hydra configuration system...")
    try:
        initialize_config_dir(config_dir=CONFIG_DIR, version_base=None)
    except Exception as e:
        print(f"❌ ERROR initializing Hydra: {e}")
        import traceback
        traceback.print_exc()
        return

    # Build configuration overrides
    print("Building configuration...")
    overrides = build_config_overrides(episode)

    try:
        # Compose configuration (same as command-line args)
        cfg = compose(config_name="experiments/vlfm_objectnav_hm3d", overrides=overrides)

        # Apply patches (same as run.py)
        cfg = patch_config(cfg)

        # Remove semantic sensor if present (same as run.py)
        with read_write(cfg):
            try:
                cfg.habitat.simulator.agents.main_agent.sim_sensors.pop("semantic_sensor")
            except KeyError:
                pass

    except Exception as e:
        print(f"❌ ERROR building configuration: {e}")
        import traceback
        traceback.print_exc()
        GlobalHydra.instance().clear()
        return

    # Run episode using original execute_exp (handles everything!)
    print("\n" + "=" * 80)
    print("STARTING EPISODE")
    print("=" * 80 + "\n")

    try:
        execute_exp(cfg, "eval")

        print("\n" + "=" * 80)
        print("✓ EPISODE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nResults saved to: {OUTPUT_DIR}")
        print("Check for video file with metrics in filename!\n")

    except KeyboardInterrupt:
        print("\n⚠ Episode interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR during episode execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        GlobalHydra.instance().clear()
        print("✓ Cleanup completed")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_episode()
# ltl/top.py
"""
Clean, simple VLFM runner without trainer complexity.
Direct control over scene, target, and start position.
"""

import os
import sys
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import cv2
import gzip
import json

# Setup paths
PROJECT_ROOT = "/home/dylan/PycharmProjects/VLM_LTL/vlfm"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Core imports
from habitat import Env, get_config
from habitat.config import read_write
import habitat_sim
import quaternion

# VLFM imports - FIXED PATHS
from vlfm.policy.itm_policy import ITMPolicyV2
from habitat_baselines.common.baseline_registry import baseline_registry

# ============================================================================
# CONFIGURATION - Change these to control the episode
# ============================================================================

SCENE_NAME = "4ok3usBNeis"
EPISODE_INDEX = 0  # Load this episode (for scene geometry/navmesh)

# Manual overrides (None = use episode defaults)
TARGET_OBJECT_OVERRIDE: Optional[str] = "chair"  # "chair", "bed", "plant", "toilet", "tv", "sofa"
START_POSITION_OVERRIDE: Optional[List[float]] = None  # [x, y, z]
START_ROTATION_OVERRIDE: Optional[List[float]] = None  # [qx, qy, qz, qw]

# Episode settings
MAX_STEPS = 500
VIDEO_FPS = 10
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ltl", "outputs")

# ============================================================================
# PATHS
# ============================================================================

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_DIR = os.path.join(DATA_DIR, "datasets/objectnav/hm3d/v1/val")
SCENE_DATASET_DIR = os.path.join(DATA_DIR, "scene_datasets")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config/experiments/vlfm_objectnav_hm3d.yaml")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_episode_info(scene_name: str, episode_index: int) -> dict:
    """Load episode from JSON."""
    scene_file = os.path.join(DATASET_DIR, f"content/{scene_name}.json.gz")

    if not os.path.exists(scene_file):
        scene_file = os.path.join(DATASET_DIR, f"content/{scene_name}.json")

    if scene_file.endswith('.gz'):
        with gzip.open(scene_file, 'rt') as f:
            data = json.load(f)
    else:
        with open(scene_file, 'r') as f:
            data = json.load(f)

    return data['episodes'][episode_index]


def create_minimal_config(episode: dict) -> Any:
    """Create minimal Habitat config."""
    config = get_config(CONFIG_PATH)

    with read_write(config):
        # Dataset
        config.habitat.dataset.split = "val"
        config.habitat.dataset.content_scenes = [SCENE_NAME]
        config.habitat.dataset.data_path = os.path.join(DATASET_DIR, "val.json.gz")
        config.habitat.dataset.scenes_dir = SCENE_DATASET_DIR

        # Simulator - DON'T set scene here, it will be loaded from episode
        config.habitat.simulator.scene_dataset = os.path.join(
            SCENE_DATASET_DIR, "hm3d_annotated_basis.scene_dataset_config.json"
        )
        # Remove this line - it creates the double path:
        # config.habitat.simulator.scene = episode['scene_id']

        # Remove semantic sensor
        try:
            config.habitat.simulator.agents.main_agent.sim_sensors.pop("semantic_sensor")
        except KeyError:
            pass

        # Environment
        config.habitat.environment.max_episode_steps = MAX_STEPS

    return config


def create_video_writer(output_path: str, frame_size: tuple, fps: int):
    """Create OpenCV video writer."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def generate_visualization_frame(observations: Dict, info: Dict, policy_info: Dict,
                                 target: str, step: int) -> np.ndarray:
    """Create single visualization frame combining all views."""
    frames = []

    # RGB view
    if 'rgb' in observations:
        rgb = observations['rgb'].copy()
        cv2.putText(rgb, f"Step: {step}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(rgb, f"Target: {target}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        frames.append(rgb)

    # Depth view
    if 'depth' in observations:
        depth = observations['depth']
        depth_vis = (depth / depth.max() * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        frames.append(depth_color)

    # Value map (from policy)
    if 'value_map' in policy_info:
        value_map = policy_info['value_map']
        if len(frames) > 0:
            h, w = frames[0].shape[:2]
            value_map = cv2.resize(value_map, (w, h))
        frames.append(value_map)

    # Top-down map
    if 'frontier_exploration_map' in info:
        top_down = info['frontier_exploration_map']
        if len(frames) > 0:
            h, w = frames[0].shape[:2]
            if len(top_down.shape) == 2:
                top_down = cv2.cvtColor(top_down, cv2.COLOR_GRAY2BGR)
            top_down = cv2.resize(top_down, (w, h))
        frames.append(top_down)

    # Combine frames
    if len(frames) == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    if len(frames) <= 2:
        combined = np.hstack(frames)
    else:
        # 2x2 grid
        top_row = np.hstack(frames[:2])
        bottom_frames = frames[2:4] if len(frames) >= 4 else [frames[2], frames[2]]
        bottom_row = np.hstack(bottom_frames)
        combined = np.vstack([top_row, bottom_row])

    return combined


def print_episode_info(episode: dict, has_overrides: bool = False):
    """Print episode information."""
    print("\n" + "=" * 80)
    if has_overrides:
        print("ORIGINAL EPISODE INFO (before overrides)")
    else:
        print("EPISODE INFO")
    print("=" * 80)
    print(f"Episode ID: {episode['episode_id']}")
    print(f"Scene: {episode['scene_id']}")
    print(f"Start Position: {episode['start_position']}")
    print(f"Start Rotation: {episode['start_rotation']}")
    print(f"Target Object: {episode['object_category']}")

    if 'info' in episode:
        print(f"\nPath Info:")
        for k, v in episode['info'].items():
            print(f"  {k}: {v}")

    print("=" * 80 + "\n")


def apply_overrides(env: Env, episode: dict):
    """Apply manual overrides to environment after reset."""
    changes = []

    # Override target
    if TARGET_OBJECT_OVERRIDE is not None:
        old = env.current_episode.object_category
        env.current_episode.object_category = TARGET_OBJECT_OVERRIDE
        episode['object_category'] = TARGET_OBJECT_OVERRIDE
        changes.append(f"Target: {old} → {TARGET_OBJECT_OVERRIDE}")

    # Override position
    if START_POSITION_OVERRIDE is not None:
        agent = env.sim.get_agent(0)
        agent_state = agent.get_state()
        old_pos = list(agent_state.position)
        agent_state.position = np.array(START_POSITION_OVERRIDE, dtype=np.float32)
        agent.set_state(agent_state)
        changes.append(f"Position: {old_pos} → {START_POSITION_OVERRIDE}")

    # Override rotation
    if START_ROTATION_OVERRIDE is not None:
        agent = env.sim.get_agent(0)
        agent_state = agent.get_state()
        old_rot = agent_state.rotation
        agent_state.rotation = quaternion.from_float_array(START_ROTATION_OVERRIDE)
        agent.set_state(agent_state)
        changes.append(f"Rotation: {old_rot} → {START_ROTATION_OVERRIDE}")

    if changes:
        print("\n" + "=" * 80)
        print("APPLIED OVERRIDES:")
        print("=" * 80)
        for change in changes:
            print(f"  ✓ {change}")
        print("=" * 80 + "\n")


def run_episode():
    """Main function - runs single episode with VLFM."""

    print("\n" + "=" * 80)
    print("VLFM CLEAN RUNNER")
    print("=" * 80 + "\n")

    # Check prerequisites
    if not os.path.exists(os.path.join(DATA_DIR, "dummy_policy.pth")):
        print("❌ ERROR: Run 'python -m vlfm.utils.generate_dummy_policy' first")
        return

    # Load episode
    print(f"Loading episode {EPISODE_INDEX} from scene {SCENE_NAME}...")
    episode = load_episode_info(SCENE_NAME, EPISODE_INDEX)

    has_overrides = any([TARGET_OBJECT_OVERRIDE, START_POSITION_OVERRIDE, START_ROTATION_OVERRIDE])
    print_episode_info(episode, has_overrides)

    # Create config
    print("Creating configuration...")
    config = create_minimal_config(episode)

    # Create environment
    print("Initializing environment...")
    env = Env(config=config)
    observations = env.reset()

    # Apply overrides
    if has_overrides:
        apply_overrides(env, episode)
        # Get fresh observations after teleport
        observations = env.sim.get_sensor_observations()

    target_object = episode['object_category']

    # Create policy using habitat_baselines registry
    # Replace the policy initialization section with this simpler version:

    # Create policy using habitat_baselines registry
    print(f"Initializing ITMPolicy for target: {target_object}...")

    # Get policy class from registry
    policy_class = baseline_registry.get_policy("HabitatITMPolicyV2")

    # Create policy directly
    policy = policy_class.from_config(
        config=config.habitat_baselines.rl.policy,
        observation_space=env.observation_space,
        action_space=env.action_space,
        orig_action_space=env.action_space,
        agent_name="agent_0"
    )

    # Initialize policy through agent
    policy = policy_class.from_config(
        config=config.habitat_baselines.rl.policy,
        observation_space=env.observation_space,
        action_space=env.action_space,
        orig_action_space=env.action_space,
        agent_name="agent_0"
    )

    # Setup video
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_path = os.path.join(
        OUTPUT_DIR,
        f"episode_{SCENE_NAME}_{target_object}_{EPISODE_INDEX}.mp4"
    )
    video_writer = None
    frames = []

    # Initialize policy state
    hidden_states = torch.zeros(1, *policy.hidden_state_shape)
    prev_actions = torch.zeros(1, 1, dtype=torch.long)
    masks = torch.ones(1, 1, dtype=torch.bool)

    # Convert observations to batch format
    batch = {k: torch.from_numpy(v).unsqueeze(0) for k, v in observations.items()}

    print("\n" + "=" * 80)
    print("RUNNING EPISODE")
    print("=" * 80 + "\n")

    step = 0
    done = False

    try:
        while not done and step < MAX_STEPS:
            # Get action from policy
            with torch.no_grad():
                action_data = policy.act(
                    batch,
                    hidden_states,
                    prev_actions,
                    masks,
                    deterministic=False
                )

            # Extract policy info for visualization
            policy_info = {}
            if hasattr(action_data, 'policy_info') and len(action_data.policy_info) > 0:
                policy_info = action_data.policy_info[0]

            # Get metrics
            info = env.get_metrics()

            # Generate visualization frame
            frame = generate_visualization_frame(
                observations, info, policy_info, target_object, step
            )
            frames.append(frame)

            # Initialize video writer with actual frame size
            if video_writer is None and len(frames) > 0:
                h, w = frame.shape[:2]
                video_writer = create_video_writer(video_path, (w, h), VIDEO_FPS)

            # Print progress
            if step % 10 == 0:
                dist = info.get('distance_to_goal', 'N/A')
                print(f"Step {step}/{MAX_STEPS} | Distance: {dist}")

            # Step environment
            action = action_data.actions[0].item()
            observations = env.step(action)

            # Update policy state
            hidden_states = action_data.rnn_hidden_states
            prev_actions = action_data.actions

            # Convert new observations to batch
            batch = {k: torch.from_numpy(v).unsqueeze(0) for k, v in observations.items()}

            # Check if done
            done = env.episode_over
            step += 1

        # Get final metrics
        metrics = env.get_metrics()

        print("\n" + "=" * 80)
        print("EPISODE COMPLETE")
        print("=" * 80)
        print(f"Steps: {step}")
        print(f"Success: {metrics.get('success', 0)}")
        print(f"SPL: {metrics.get('spl', 0):.3f}")
        print(f"Distance to goal: {metrics.get('distance_to_goal', 0):.3f}")
        print("=" * 80 + "\n")

        # Save video
        if video_writer is not None and len(frames) > 0:
            print(f"Saving video ({len(frames)} frames)...")
            for frame in frames:
                video_writer.write(frame)
            video_writer.release()

            # Rename with metrics
            success = int(metrics.get('success', 0))
            spl = metrics.get('spl', 0)
            final_name = os.path.join(
                OUTPUT_DIR,
                f"episode_{SCENE_NAME}_{target_object}_success={success}_spl={spl:.2f}.mp4"
            )
            os.rename(video_path, final_name)
            print(f"✓ Video saved: {final_name}\n")

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if video_writer is not None:
            video_writer.release()
        env.close()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    # Import registration modules
    import frontier_exploration  # noqa
    import vlfm.measurements.traveled_stairs  # noqa
    import vlfm.obs_transformers.resize  # noqa
    import vlfm.policy.action_replay_policy  # noqa
    import vlfm.policy.habitat_policies  # noqa

    run_episode()
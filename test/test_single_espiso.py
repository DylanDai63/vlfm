# run_single_episode.py
"""
Simple script to run VLFM on a single episode.
Just hit 'Run' in PyCharm!

Before running:
1. Start VLM servers: ./scripts/launch_vlm_servers.sh
2. Wait 2-3 minutes for models to load
"""

import os
import sys

# Go up two levels: test -> vlfm -> VLM_LTL
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(project_root, 'vlfm'))  # Add vlfm to path for imports

import json
import gzip
import socket
from pathlib import Path
from datetime import datetime

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from habitat.config.default import patch_config
from habitat.config import read_write
from habitat_baselines.run import execute_exp

# Import to register components
import vlfm.measurements.traveled_stairs
import vlfm.obs_transformers.resize
import vlfm.policy.action_replay_policy
import vlfm.policy.habitat_policies
import vlfm.utils.vlfm_trainer
import frontier_exploration

# ============================================================================
# CONFIGURATION - Change these as needed
# ============================================================================

EPISODE_CONFIG = {
    'scene_name': '4ok3usBNeis',  # Which scene to use
    'episode_id': None,  # None = list and choose, or set number (0, 1, 2...)
}

VIDEO_CONFIG = {
    'generate': True,
    'fps': 10,
}

MAP_CONFIG = {
    'resolution': 1024,
    'visibility_distance': 5.0,
    'fov': 90,
}

NAVIGATION_CONFIG = {
    'max_steps': 500,
}

OUTPUT_DIR = 'test_results'


# ============================================================================
# Helper Functions
# ============================================================================

def check_vlm_servers():
    """Check if VLM servers are running on ports 12181-12184"""
    # ports = [12181, 12182, 12183, 12184]
    # server_names = ['GroundingDINO', 'BLIP-2', 'MobileSAM', 'YOLOv7']
    ports = [ 12182, 12183, 12184]
    server_names = [ 'BLIP-2', 'MobileSAM', 'YOLOv7']

    all_running = True
    for port, name in zip(ports, server_names):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()

        if result != 0:
            print(f"❌ {name} (port {port}) is NOT running")
            all_running = False
        else:
            print(f"✅ {name} (port {port}) is running")

    if not all_running:
        print("\n⚠️  Please start VLM servers first:")
        print("   ./scripts/launch_vlm_servers.sh")
        print("   Then wait 2-3 minutes for models to load\n")
        sys.exit(1)

    print("✅ All VLM servers are running!\n")


def load_episodes(scene_name):
    """Load episodes for a specific scene"""
    episodes_file = f'/home/dylan/PycharmProjects/VLM_LTL/vlfm/data/datasets/objectnav/hm3d/v1/val/content/{scene_name}.json.gz'

    if not os.path.exists(episodes_file):
        print(f"❌ Scene '{scene_name}' not found!")
        print(f"   Looking for: {episodes_file}")

        # List available scenes
        content_dir = '/home/dylan/PycharmProjects/VLM_LTL/vlfm/data/datasets_pp/objectnav/hm3d/v1/val/content'
        if os.path.exists(content_dir):
            available = [f.replace('.json.gz', '') for f in os.listdir(content_dir)
                         if f.endswith('.json.gz')]
            print(f"\n📁 Available scenes: {available[:10]}...")
        sys.exit(1)

    with gzip.open(episodes_file, 'rt') as f:
        data = json.load(f)

    return data['episodes']


def select_episode(scene_name, episode_id=None):
    """Select episode from scene"""
    episodes = load_episodes(scene_name)

    if episode_id is None:
        # List episodes and let user choose
        print(f"\n📋 Episodes in scene '{scene_name}':")
        print("-" * 80)
        for i, ep in enumerate(episodes[:10]):  # Show first 10
            print(f"  [{i}] Target: {ep['object_category']:8s} | "
                  f"Start: {ep['start_position']} | "
                  f"Distance: {ep['info']['geodesic_distance']:.2f}m")

        if len(episodes) > 10:
            print(f"  ... and {len(episodes) - 10} more episodes")

        print("-" * 80)
        choice = input(f"\nSelect episode [0-{len(episodes) - 1}] or press Enter for 0: ")
        episode_id = int(choice) if choice.strip() else 0

    if episode_id >= len(episodes):
        print(f"❌ Episode {episode_id} not found! Scene has {len(episodes)} episodes.")
        sys.exit(1)

    selected = episodes[episode_id]
    print(f"\n🎯 Selected Episode {episode_id}:")
    print(f"   Target: {selected['object_category']}")
    print(f"   Start position: {selected['start_position']}")
    print(f"   Optimal distance: {selected['info']['geodesic_distance']:.2f}m\n")

    return selected


def setup_output_dir():
    """Create output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(OUTPUT_DIR) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("=" * 80)
    print("VLFM Single Episode Runner")
    print("=" * 80 + "\n")

    # 1. Check prerequisites
    # if not os.path.isfile("data/dummy_policy.pth"):
    #     print("❌ Dummy policy not found! Run: python -m vlfm.utils.generate_dummy_policy")
    #     sys.exit(1)

    check_vlm_servers()

    # 2. Select episode
    episode = select_episode(EPISODE_CONFIG['scene_name'], EPISODE_CONFIG['episode_id'])

    # 3. Setup output directory
    output_path = setup_output_dir()
    video_path = output_path / "videos"
    video_path.mkdir(exist_ok=True)

    print(f"📁 Output directory: {output_path}\n")

    # 4. Load and configure Hydra
    config_dir = os.path.join(project_root, "vlfm/config")  # Change this line
    config_dir = os.path.abspath(config_dir)

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="experiments/vlfm_objectnav_hm3d",
            overrides=[
                # Episode settings
                "habitat_baselines.evaluate=True",
                "habitat_baselines.test_episode_count=1",
                "habitat_baselines.num_environments=1",
                f"habitat.dataset.content_scenes=[{EPISODE_CONFIG['scene_name']}]",

                # Video settings
                f"habitat_baselines.video_dir={str(video_path)}",
                f"habitat_baselines.video_fps={VIDEO_CONFIG['fps']}",
                "+habitat_baselines.eval.video_option=['disk']" if VIDEO_CONFIG['generate'] else "",

                # Navigation settings
                f"habitat.environment.max_episode_steps={NAVIGATION_CONFIG['max_steps']}",
                f"habitat.task.measurements.frontier_exploration_map.max_episode_steps={NAVIGATION_CONFIG['max_steps'] * 2}",

                # Map settings
                f"habitat.task.measurements.frontier_exploration_map.map_resolution={MAP_CONFIG['resolution']}",
                f"habitat.task.measurements.frontier_exploration_map.fog_of_war.visibility_dist={MAP_CONFIG['visibility_distance']}",
                f"habitat.task.measurements.frontier_exploration_map.fog_of_war.fov={MAP_CONFIG['fov']}",

                # Visualization
                "habitat.task.measurements.frontier_exploration_map.draw_source=True",
                "habitat.task.measurements.frontier_exploration_map.draw_border=True",
                "habitat.task.measurements.frontier_exploration_map.draw_shortest_path=True",
                "habitat.task.measurements.frontier_exploration_map.draw_waypoints=True",
                "habitat.task.measurements.frontier_exploration_map.draw_goal_positions=True",
                "habitat.task.measurements.frontier_exploration_map.fog_of_war.draw=True",

                # Logging
                "habitat_baselines.verbose=True",
            ]
        )

        # Apply patches
        cfg = patch_config(cfg)
        with read_write(cfg):
            # Remove semantic sensor (we use vision models instead)
            try:
                cfg.habitat.simulator.agents.main_agent.sim_sensors.pop("semantic_sensor")
            except KeyError:
                pass

        # 5. Run episode
        print("🚀 Starting navigation...\n")
        print("=" * 80)

        execute_exp(cfg, "eval")

        print("=" * 80)
        print(f"\n✅ Done! Results saved to: {output_path}")
        print(f"   Videos: {video_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
View episode details from Habitat ObjectNav dataset
"""

import gzip
import json
from pathlib import Path
from collections import Counter

# Get the correct paths
project_root = Path(__file__).parent.parent
content_dir = project_root / "data/datasets/objectnav/hm3d/v1/val/content"
val_file = project_root / "data/datasets/objectnav/hm3d/v1/val/val.json.gz"

print("🔍 Checking dataset structure...\n")

# Check if content directory exists
if not content_dir.exists():
    print(f"❌ Content directory not found: {content_dir}")
    exit(1)

print(f"✅ Content directory: {content_dir}\n")

# List all .json.gz files
scene_files = list(content_dir.glob("*.json.gz"))
print(f"📁 Found {len(scene_files)} scene files\n")

# Try to load val.json.gz first
all_episodes = []

if val_file.exists():
    print(f"📂 Loading main dataset file: val.json.gz")
    with gzip.open(val_file, 'rt') as f:
        data = json.load(f)
        all_episodes = data.get('episodes', [])
    print(f"   Episodes in val.json.gz: {len(all_episodes)}")

# If val.json.gz is empty or doesn't exist, load from scene files
if len(all_episodes) == 0:
    print("\n📦 Loading episodes from individual scene files...")
    for scene_file in sorted(scene_files)[:5]:  # Load first 5 scenes as sample
        try:
            with gzip.open(scene_file, 'rt') as f:
                scene_data = json.load(f)
                episodes = scene_data.get('episodes', [])
                if episodes:
                    all_episodes.extend(episodes)
                    print(f"   ✓ {scene_file.name}: {len(episodes)} episodes")
        except Exception as e:
            print(f"   ✗ {scene_file.name}: Error - {e}")

if len(all_episodes) == 0:
    print("\n❌ No episodes found!")
    print("\nTrying to inspect one scene file...")

    # Debug: Look at structure of first file
    if scene_files:
        first_file = scene_files[0]
        print(f"\n🔍 Inspecting: {first_file.name}")
        with gzip.open(first_file, 'rt') as f:
            data = json.load(f)
            print(f"   Keys in file: {list(data.keys())}")
            if 'episodes' in data:
                print(f"   Episodes: {len(data['episodes'])}")
            else:
                print(f"   No 'episodes' key found")
    exit(1)

print(f"\n{'=' * 80}")
print(f"Total Episodes Loaded: {len(all_episodes)}")
print(f"{'=' * 80}\n")

# Show first 10 episodes
print("📋 First 10 Episodes:\n")
print("-" * 80)

for i, ep in enumerate(all_episodes[:10]):
    # Extract scene name
    scene_id = ep.get('scene_id', 'Unknown')
    if '/' in scene_id:
        scene_name = scene_id.split('/')[-2].split('-')[-1]
    else:
        scene_name = scene_id

    episode_id = ep.get('episode_id', 'N/A')
    object_category = ep.get('object_category', 'N/A')
    start_pos = ep.get('start_position', [0, 0, 0])

    goals = ep.get('goals', [])
    if goals:
        goal_pos = goals[0].get('position', [0, 0, 0])
        goal_radius = goals[0].get('radius', 1.0)
    else:
        goal_pos = [0, 0, 0]
        goal_radius = 1.0

    print(f"Episode {i + 1}:")
    print(f"  Episode ID: {episode_id}")
    print(f"  Scene: {scene_name}")
    print(f"  🎯 Task: Find '{object_category}'")
    print(f"  📍 Start: [{start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}]")
    print(f"  🏁 Goal: [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f}] (radius: {goal_radius}m)")
    print("-" * 80)

# Object statistics
print("\n📊 Object Category Distribution:\n")
object_counts = Counter(ep.get('object_category', 'Unknown') for ep in all_episodes)

for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(all_episodes)) * 100
    bar = "█" * min(int(percentage / 2), 50)
    print(f"  {obj:20s} {count:4d} episodes ({percentage:5.1f}%) {bar}")

print(f"\n  {'TOTAL':20s} {len(all_episodes):4d} episodes")

# Scene statistics
print("\n🏠 Scenes with Episodes:\n")
scene_counts = Counter()
for ep in all_episodes:
    scene_id = ep.get('scene_id', 'Unknown')
    if '/' in scene_id:
        scene_name = scene_id.split('/')[-2].split('-')[-1]
    else:
        scene_name = scene_id
    scene_counts[scene_name] += 1

for i, (scene, count) in enumerate(scene_counts.most_common(10), 1):
    print(f"  {i:2d}. {scene:15s} → {count:3d} episodes")

print(f"\n{'=' * 80}")
print("\n💡 To run specific episodes:")
print("  • habitat_baselines.test_episode_count=1")
print("  • habitat.dataset.content_scenes='[\"SCENE_NAME\"]'")
print("\n")
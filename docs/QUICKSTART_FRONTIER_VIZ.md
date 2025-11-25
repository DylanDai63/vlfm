# Quick Start: Visualizing Frontiers in Camera View

## 🚀 Run with Frontier Visualization

### Option 1: Use the provided script

```bash
./scripts/eval_with_frontier_viz.sh
```

### Option 2: Run with your existing command

Just run your existing command as-is! The frontier visualization is now **automatically included** when you run evaluation:

```bash
python -m vlfm.run \
  habitat_baselines.evaluate=True \
  habitat_baselines.test_episode_count=1 \
  habitat_baselines.num_environments=1 \
  habitat_baselines.video_dir=test_videos \
  habitat_baselines.video_fps=10 \
  habitat_baselines.verbose=True \
  '+habitat_baselines.eval.video_option=["disk"]' \
  habitat.dataset.split=val \
  'habitat.dataset.content_scenes=["4ok3usBNeis"]' \
  habitat.environment.max_episode_steps=500 \
  habitat.task.measurements.frontier_exploration_map.max_episode_steps=1000 \
  habitat.task.measurements.frontier_exploration_map.map_resolution=1024 \
  habitat.task.measurements.frontier_exploration_map.draw_source=True \
  habitat.task.measurements.frontier_exploration_map.draw_border=True \
  habitat.task.measurements.frontier_exploration_map.draw_shortest_path=True \
  habitat.task.measurements.frontier_exploration_map.draw_waypoints=True \
  habitat.task.measurements.frontier_exploration_map.draw_goal_positions=True \
  habitat.task.measurements.frontier_exploration_map.fog_of_war.draw=True \
  habitat.task.measurements.frontier_exploration_map.fog_of_war.visibility_dist=5.0 \
  habitat.task.measurements.frontier_exploration_map.fog_of_war.fov=90
```

## 📹 What You'll See in the Video

The generated video will now include:

| Frame Name | Description |
|------------|-------------|
| `annotated_rgb` | RGB with detected objects (existing) |
| **`rgb_with_frontiers`** | **RGB with frontiers projected onto camera (NEW!)** |
| `annotated_depth` | Depth with object masks (existing) |
| `obstacle_map` | Top-down map with obstacles and frontiers (existing) |
| `value_map` | Language-grounded value map (existing) |

## 🎨 Visualization Features

The `rgb_with_frontiers` frame shows:

- **🔵 Colored Circles**: Each frontier is marked with a unique color
- **🔢 Labels**: Frontier indices (F0, F1, F2, ...)
- **➡️ Direction Arrows**: Arrows from image center pointing to each frontier
- **⭐ Selected Frontier**: The currently selected frontier is highlighted in **yellow**
- **📊 Info Overlay**: Shows "Frontiers: X/Y visible" at the top

## 🎯 Example Output

When a frontier is visible in the camera view, you'll see:
- A circle marking its approximate location in the image
- An arrow pointing from the center of the view to that frontier
- A label like "F0", "F1", etc.
- If it's the selected frontier (where the robot is heading), it'll be yellow

## ⚙️ Customization

### Change Camera Height

If your camera is at a different height, modify `vlfm/policy/itm_policy.py` line 206:

```python
point_height=1.5,  # Change to your camera height in meters
```

### Disable Frontier Visualization

To turn off the frontier visualization but keep other visualizations:

```bash
# Add this to your command:
habitat_baselines.rl.policy.compute_frontiers=False
```

Or set `visualize=False` in the policy config to disable all visualizations.

## 📍 Where Videos are Saved

Videos are saved to the directory specified by `habitat_baselines.video_dir`:
- Default: `video_dir/`
- In the example script: `test_videos_with_frontiers/`

Each episode will generate a `.mp4` file containing all the visualization frames stacked together.

## 🔍 Troubleshooting

**Q: I don't see frontiers in the camera view**
- Frontiers might be behind the camera or outside the field of view
- Try running for more steps so the robot explores and detects frontiers
- Check the `obstacle_map` frame to see where frontiers are in the top-down view

**Q: The projection looks wrong**
- Verify the camera height parameter (line 206 in `itm_policy.py`)
- Check that your camera intrinsics (fx, fy) are correct

**Q: Video is not being saved**
- Make sure `'+habitat_baselines.eval.video_option=["disk"]'` is in your command
- Check that the video_dir exists and is writable

## 📚 More Information

For detailed usage and advanced customization, see:
- `docs/FRONTIER_PROJECTION_USAGE.md` - Comprehensive usage guide
- `scripts/visualize_frontiers_in_camera.py` - Standalone visualization function
- `test/test_frontier_projection.py` - Unit tests and examples

## 🎉 That's It!

The frontier visualization is now integrated into your existing workflow. Just run your evaluation command and the annotated RGB frames will automatically be included in your videos!

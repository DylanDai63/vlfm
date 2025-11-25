#!/bin/bash

# Example script to run VLFM evaluation with frontier-to-camera projection visualization
# The annotated RGB frames with frontiers will be automatically included in the video output

python -m vlfm.run \
  habitat_baselines.evaluate=True \
  habitat_baselines.test_episode_count=1 \
  habitat_baselines.num_environments=1 \
  habitat_baselines.video_dir=test_videos_with_frontiers \
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

echo ""
echo "✅ Evaluation complete!"
echo "📹 Videos saved to: test_videos_with_frontiers/"
echo "🎯 The videos will include:"
echo "   - annotated_rgb: RGB with object detections"
echo "   - rgb_with_frontiers: RGB with frontiers projected onto camera view (NEW!)"
echo "   - obstacle_map: Top-down obstacle map with frontiers"
echo "   - value_map: Language-grounded value map"

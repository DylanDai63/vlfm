# Frontier-to-Camera Projection - Usage Guide

This guide shows you how to project frontiers from the 2D top-down map back onto RGB camera frames, so you can see which direction in your camera view corresponds to each detected frontier.

## Table of Contents
1. [Quick Integration into Existing Policy](#quick-integration)
2. [Standalone Usage](#standalone-usage)
3. [Real-World Robot Example](#real-world-example)
4. [Advanced Usage](#advanced-usage)

---

## Quick Integration

### Option 1: Add to Your Policy's Visualization

If you're already using an ITM policy (like `ITMPolicyV2` or `ITMPolicyV3`), add this to the `_get_policy_info()` method:

```python
from vlfm.utils.geometry_utils import project_map_points_to_image
from scripts.visualize_frontiers_in_camera import visualize_frontiers_in_camera

def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
    # ... existing code ...

    # Get the first RGB image and camera parameters
    rgb, depth, tf_camera_to_episodic, min_depth, max_depth, fx, fy = \
        self._observations_cache["object_map_rgbd"][0]

    # Get image dimensions
    height, width = rgb.shape[:2]

    # Visualize frontiers on the RGB frame
    rgb_with_frontiers = visualize_frontiers_in_camera(
        rgb_image=rgb,
        obstacle_map=self._obstacle_map,
        tf_camera_to_episodic=tf_camera_to_episodic,
        fx=fx,
        fy=fy,
        camera_height=1.5,  # Adjust based on your camera height
    )

    # Add to policy info for visualization
    policy_info["rgb_with_frontiers"] = rgb_with_frontiers

    return policy_info
```

### Option 2: Just Get the Pixel Coordinates

If you only need the pixel coordinates without visualization:

```python
from vlfm.utils.geometry_utils import project_map_points_to_image

# In your policy's act() or _explore() method:
frontiers_xy = self._obstacle_map.frontiers  # (N, 2) array

rgb, depth, tf_camera_to_episodic, _, _, fx, fy = \
    self._observations_cache["object_map_rgbd"][0]

height, width = rgb.shape[:2]

# Project frontiers to image coordinates
pixel_coords, valid_mask = project_map_points_to_image(
    map_points_xy=frontiers_xy,
    tf_camera_to_episodic=tf_camera_to_episodic,
    fx=fx,
    fy=fy,
    image_width=width,
    image_height=height,
    point_height=1.5,  # Camera height in meters
)

# Get only visible frontiers
visible_frontier_pixels = pixel_coords[valid_mask]
visible_frontier_indices = np.where(valid_mask)[0]

print(f"Found {len(visible_frontier_pixels)} visible frontiers out of {len(frontiers_xy)}")

# Now you can use these pixel coordinates for:
# - Drawing on the image
# - Checking if a frontier overlaps with detected objects
# - Computing visual features at frontier locations
```

---

## Standalone Usage

### Example 1: Debugging Frontier Detection

Create a standalone script to visualize frontiers during navigation:

```python
#!/usr/bin/env python3
import cv2
import numpy as np
from vlfm.mapping.obstacle_map import ObstacleMap
from scripts.visualize_frontiers_in_camera import visualize_frontiers_in_camera

# Initialize obstacle map
obstacle_map = ObstacleMap(
    min_height=-0.5,
    max_height=2.0,
    agent_radius=0.18,
)

# In your navigation loop:
while True:
    # Get observations from your environment
    rgb = env.get_rgb()  # (H, W, 3) numpy array
    depth = env.get_depth()  # (H, W) normalized to [0, 1]
    tf_camera_to_episodic = env.get_camera_pose()  # 4x4 matrix
    fx, fy = env.get_focal_lengths()  # Camera intrinsics

    # Update obstacle map (detects frontiers)
    obstacle_map.update_map(
        depth=depth,
        tf_camera_to_episodic=tf_camera_to_episodic,
        min_depth=0.5,
        max_depth=5.0,
        fx=fx,
        fy=fy,
        topdown_fov=np.deg2rad(79),
    )

    # Visualize frontiers on RGB frame
    annotated_rgb = visualize_frontiers_in_camera(
        rgb_image=rgb,
        obstacle_map=obstacle_map,
        tf_camera_to_episodic=tf_camera_to_episodic,
        fx=fx,
        fy=fy,
    )

    # Display
    cv2.imshow('Camera View with Frontiers', annotated_rgb)
    cv2.imshow('Top-down Map', obstacle_map.visualize())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

### Example 2: Rank Frontiers by Visual Features

```python
from vlfm.utils.geometry_utils import project_map_points_to_image

def rank_frontiers_by_visual_saliency(
    frontiers_xy,
    rgb,
    tf_camera_to_episodic,
    fx,
    fy
):
    """
    Rank frontiers by how visually interesting they are in the camera view.
    """
    height, width = rgb.shape[:2]

    # Project to image
    pixel_coords, valid_mask = project_map_points_to_image(
        map_points_xy=frontiers_xy,
        tf_camera_to_episodic=tf_camera_to_episodic,
        fx=fx,
        fy=fy,
        image_width=width,
        image_height=height,
        point_height=1.5,
    )

    # Compute saliency scores for visible frontiers
    scores = []
    for i, is_valid in enumerate(valid_mask):
        if is_valid:
            u, v = int(pixel_coords[i, 0]), int(pixel_coords[i, 1])

            # Extract a patch around the frontier
            patch = rgb[max(0, v-25):v+25, max(0, u-25):u+25]

            # Compute saliency (e.g., color variance)
            saliency = np.std(patch) if patch.size > 0 else 0
            scores.append((i, saliency))
        else:
            scores.append((i, -1))  # Not visible

    # Sort by saliency (highest first)
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores

# Usage:
frontiers_xy = obstacle_map.frontiers
ranked_frontiers = rank_frontiers_by_visual_saliency(
    frontiers_xy, rgb, tf_camera_to_episodic, fx, fy
)

print("Top 3 most visually interesting frontiers:")
for idx, saliency in ranked_frontiers[:3]:
    print(f"  Frontier {idx}: saliency={saliency:.2f}")
```

---

## Real-World Example

### For Boston Dynamics Spot Robot

If you're using the reality deployment (`vlfm/reality/`):

```python
from vlfm.reality.robots.bd_spot_wrapper import BDSWRobot
from vlfm.mapping.obstacle_map import ObstacleMap
from scripts.visualize_frontiers_in_camera import visualize_frontiers_in_camera
import numpy as np

# Initialize robot and map
robot = BDSWRobot()
obstacle_map = ObstacleMap(min_height=0.0, max_height=2.0, agent_radius=0.3)

# Get current observations
rgb_images = robot.get_rgb_images()  # Dict of camera name -> image
depth_images = robot.get_depth_images()  # Dict of camera name -> depth

# Use hand camera for visualization
camera_name = "hand_color"
rgb = rgb_images[camera_name]
depth = depth_images[camera_name]

# Get camera pose and intrinsics
tf_camera_to_episodic = robot.get_camera_transform(camera_name)
fx, fy = robot.get_camera_intrinsics(camera_name)

# Update obstacle map
obstacle_map.update_map(
    depth=depth,
    tf_camera_to_episodic=tf_camera_to_episodic,
    min_depth=0.5,
    max_depth=5.0,
    fx=fx,
    fy=fy,
    topdown_fov=np.deg2rad(79),
)

# Visualize
annotated = visualize_frontiers_in_camera(
    rgb_image=rgb,
    obstacle_map=obstacle_map,
    tf_camera_to_episodic=tf_camera_to_episodic,
    fx=fx,
    fy=fy,
    camera_height=robot.camera_height,
)

cv2.imwrite("spot_frontiers.jpg", annotated)
```

---

## Advanced Usage

### Check if Frontier Overlaps with Detected Object

```python
from vlfm.utils.geometry_utils import project_map_points_to_image

def frontier_near_object(frontier_xy, object_bbox, rgb, tf_camera_to_episodic, fx, fy):
    """
    Check if a frontier is near a detected object in the image.

    Args:
        frontier_xy: (2,) array with frontier (x, y) in map frame
        object_bbox: [x1, y1, x2, y2] bounding box in image coordinates
        rgb: RGB image
        tf_camera_to_episodic: Camera pose
        fx, fy: Camera intrinsics

    Returns:
        True if frontier projects near the object bounding box
    """
    height, width = rgb.shape[:2]

    pixel_coords, valid_mask = project_map_points_to_image(
        map_points_xy=frontier_xy.reshape(1, 2),
        tf_camera_to_episodic=tf_camera_to_episodic,
        fx=fx,
        fy=fy,
        image_width=width,
        image_height=height,
        point_height=1.5,
    )

    if not valid_mask[0]:
        return False

    u, v = pixel_coords[0]
    x1, y1, x2, y2 = object_bbox

    # Check if frontier pixel is inside or near the bbox
    margin = 50  # pixels
    return (x1 - margin <= u <= x2 + margin and
            y1 - margin <= v <= y2 + margin)
```

### Custom Visualization

```python
import cv2
import numpy as np
from vlfm.utils.geometry_utils import project_map_points_to_image

def draw_frontier_with_value(rgb, frontiers_xy, frontier_values,
                              tf_camera_to_episodic, fx, fy):
    """
    Draw frontiers on RGB with color-coded value scores.
    """
    vis = rgb.copy()
    height, width = rgb.shape[:2]

    pixel_coords, valid_mask = project_map_points_to_image(
        map_points_xy=frontiers_xy,
        tf_camera_to_episodic=tf_camera_to_episodic,
        fx=fx,
        fy=fy,
        image_width=width,
        image_height=height,
        point_height=1.5,
    )

    # Normalize values to [0, 1]
    if len(frontier_values) > 0:
        min_val = min(frontier_values)
        max_val = max(frontier_values)
        if max_val > min_val:
            norm_values = [(v - min_val) / (max_val - min_val)
                          for v in frontier_values]
        else:
            norm_values = [0.5] * len(frontier_values)

    for i, (is_valid, (u, v)) in enumerate(zip(valid_mask, pixel_coords)):
        if not is_valid:
            continue

        # Color from blue (low value) to red (high value)
        value = norm_values[i]
        color = (
            int(255 * (1 - value)),  # Blue decreases
            0,
            int(255 * value)  # Red increases
        )

        cv2.circle(vis, (int(u), int(v)), 8, color, -1)
        cv2.putText(vis, f"{frontier_values[i]:.2f}",
                   (int(u) + 10, int(v)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return vis
```

---

## Parameters Reference

### `project_map_points_to_image()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `map_points_xy` | `np.ndarray` | (N, 2) array of map coordinates |
| `tf_camera_to_episodic` | `np.ndarray` | 4x4 transformation matrix |
| `fx`, `fy` | `float` | Camera focal lengths in pixels |
| `image_width`, `image_height` | `int` | Image dimensions |
| `point_height` | `float` | Height to project at (default: 0.0) |

**Returns:**
- `pixel_coords`: (N, 2) array of (u, v) pixel coordinates
- `valid_mask`: (N,) boolean array indicating visibility

### `visualize_frontiers_in_camera()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `rgb_image` | `np.ndarray` | RGB image (H, W, 3) |
| `obstacle_map` | `ObstacleMap` | Map with detected frontiers |
| `tf_camera_to_episodic` | `np.ndarray` | 4x4 transformation matrix |
| `fx`, `fy` | `float` | Camera focal lengths |
| `camera_height` | `float` | Camera height above ground (default: 1.5) |
| `show_labels` | `bool` | Show frontier indices (default: True) |
| `show_direction_arrows` | `bool` | Show arrows to frontiers (default: True) |

**Returns:**
- Annotated RGB image with frontiers labeled

---

## Common Issues

### 1. **No frontiers visible**
- Frontiers might be behind the camera or outside the field of view
- Check `valid_mask` to see how many are actually visible
- Try rotating the camera or moving to a different location

### 2. **Wrong projection location**
- Verify `point_height` matches your camera height
- Check that `tf_camera_to_episodic` is the correct transformation
- Ensure `fx`, `fy` are in pixels, not normalized

### 3. **Frontiers appear at wrong depth**
- Adjust `point_height` parameter to match where you expect frontiers to appear
- If frontiers are on elevated surfaces, increase `point_height`

---

## Tips

1. **Use with value maps**: Combine frontier projection with ITM scores to see which visual regions have high semantic value

2. **Debug visualization**: Display both the top-down map and camera view side-by-side to verify projection accuracy

3. **Filter by distance**: Only project frontiers within a certain distance to reduce clutter

4. **Multi-camera**: If using multiple cameras (like on Spot), project frontiers onto each camera view separately

---

## Next Steps

- Integrate into your custom policy for real-time visualization
- Combine with object detection to check frontier-object relationships
- Use for dataset annotation or analysis
- Extend for multi-floor or 3D frontier detection

For more examples, see:
- `scripts/visualize_frontiers_in_camera.py` - Full visualization implementation
- `test/test_frontier_projection.py` - Unit tests with various scenarios
- `vlfm/utils/geometry_utils.py:274` - Core projection function

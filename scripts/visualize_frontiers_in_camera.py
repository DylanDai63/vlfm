#!/usr/bin/env python3
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Example script demonstrating how to project frontiers from the 2D map
back onto the RGB camera frame for visualization.

This allows you to see which direction in the camera view corresponds
to each detected frontier.
"""

import cv2
import numpy as np

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.utils.geometry_utils import project_map_points_to_image


def visualize_frontiers_in_camera(
    rgb_image: np.ndarray,
    obstacle_map: ObstacleMap,
    tf_camera_to_episodic: np.ndarray,
    fx: float,
    fy: float,
    camera_height: float = 1.5,
    show_labels: bool = True,
    show_direction_arrows: bool = True,
) -> np.ndarray:
    """
    Projects frontiers from the obstacle map onto the RGB camera image.

    Args:
        rgb_image: RGB camera image of shape (H, W, 3)
        obstacle_map: ObstacleMap instance with detected frontiers
        tf_camera_to_episodic: 4x4 transformation matrix from camera to episodic frame
        fx: Camera focal length in x direction (pixels)
        fy: Camera focal length in y direction (pixels)
        camera_height: Height of the camera above ground in meters
        show_labels: Whether to show frontier index labels
        show_direction_arrows: Whether to show direction arrows from image center

    Returns:
        Annotated RGB image with frontiers labeled
    """
    vis_img = rgb_image.copy()
    height, width = rgb_image.shape[:2]

    # Get frontiers from obstacle map
    frontiers_xy = obstacle_map.frontiers  # (N, 2) array in episodic frame

    if len(frontiers_xy) == 0:
        # No frontiers to visualize
        cv2.putText(
            vis_img,
            "No frontiers detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return vis_img

    # Project frontiers to image coordinates
    pixel_coords, valid_mask = project_map_points_to_image(
        map_points_xy=frontiers_xy,
        tf_camera_to_episodic=tf_camera_to_episodic,
        fx=fx,
        fy=fy,
        image_width=width,
        image_height=height,
        point_height=camera_height,
    )

    # Filter to only visible frontiers
    visible_frontiers = pixel_coords[valid_mask]
    visible_indices = np.where(valid_mask)[0]

    # Image center for direction arrows
    center_x, center_y = width // 2, height // 2

    # Draw each visible frontier
    for idx, (u, v) in enumerate(visible_frontiers):
        frontier_idx = visible_indices[idx]
        u_int, v_int = int(u), int(v)

        # Choose color based on frontier index (cycle through colors)
        colors = [
            (255, 0, 0),  # Blue
            (0, 255, 0),  # Green
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        color = colors[frontier_idx % len(colors)]

        # Draw circle at frontier location
        cv2.circle(vis_img, (u_int, v_int), 10, color, 2)
        cv2.circle(vis_img, (u_int, v_int), 3, color, -1)

        # Draw label with frontier index
        if show_labels:
            label = f"F{frontier_idx}"
            cv2.putText(
                vis_img,
                label,
                (u_int + 15, v_int - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Draw direction arrow from center to frontier
        if show_direction_arrows:
            cv2.arrowedLine(
                vis_img,
                (center_x, center_y),
                (u_int, v_int),
                color,
                2,
                tipLength=0.2,
            )

    # Add info text
    info_text = f"Frontiers: {len(visible_indices)}/{len(frontiers_xy)} visible"
    cv2.putText(
        vis_img,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        vis_img,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        1,
    )

    return vis_img


def example_usage():
    """
    Example showing how to integrate this into your navigation loop.
    """
    # Initialize obstacle map
    obstacle_map = ObstacleMap(
        min_height=-0.5,
        max_height=2.0,
        agent_radius=0.18,
        size=1000,
        pixels_per_meter=20,
    )

    # In your navigation loop, after updating the obstacle map:
    # 1. Get current RGB image and depth
    # rgb_image = env.observation['rgb']
    # depth_image = env.observation['depth']
    # tf_camera_to_episodic = env.observation['camera_pose']
    # fx, fy = env.camera_intrinsics['fx'], env.camera_intrinsics['fy']

    # 2. Update obstacle map (this detects frontiers)
    # obstacle_map.update_map(
    #     depth=depth_image,
    #     tf_camera_to_episodic=tf_camera_to_episodic,
    #     min_depth=0.5,
    #     max_depth=5.0,
    #     fx=fx,
    #     fy=fy,
    #     topdown_fov=np.deg2rad(79),
    # )

    # 3. Visualize frontiers in camera frame
    # annotated_image = visualize_frontiers_in_camera(
    #     rgb_image=rgb_image,
    #     obstacle_map=obstacle_map,
    #     tf_camera_to_episodic=tf_camera_to_episodic,
    #     fx=fx,
    #     fy=fy,
    #     camera_height=1.5,
    # )

    # 4. Display or save the annotated image
    # cv2.imshow('Frontiers in Camera View', annotated_image)
    # cv2.waitKey(1)

    print("See the example_usage() function for integration details.")
    print("\nTo use this in your code:")
    print("1. Import: from scripts.visualize_frontiers_in_camera import visualize_frontiers_in_camera")
    print("2. Call it after updating your obstacle map")
    print("3. It will return an annotated RGB image with frontiers labeled")


if __name__ == "__main__":
    example_usage()

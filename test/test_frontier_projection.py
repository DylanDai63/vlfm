#!/usr/bin/env python3
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Unit tests for projecting map points to image coordinates.
"""

import numpy as np
import pytest

from vlfm.utils.geometry_utils import project_map_points_to_image, xyz_yaw_to_tf_matrix


def test_project_point_directly_in_front():
    """Test projecting a point directly in front of the camera."""
    # Camera at origin looking along +x axis
    tf_camera_to_episodic = xyz_yaw_to_tf_matrix(np.array([0, 0, 1.5]), 0)

    # Point 5 meters in front of camera
    map_points = np.array([[5.0, 0.0]])

    # Camera parameters
    fx, fy = 320.0, 320.0  # Focal length
    width, height = 640, 480

    pixel_coords, valid_mask = project_map_points_to_image(
        map_points_xy=map_points,
        tf_camera_to_episodic=tf_camera_to_episodic,
        fx=fx,
        fy=fy,
        image_width=width,
        image_height=height,
        point_height=1.5,
    )

    # Point should be valid and at image center
    assert valid_mask[0], "Point directly in front should be visible"
    assert np.abs(pixel_coords[0, 0] - width / 2) < 1, "Point should be at horizontal center"
    assert np.abs(pixel_coords[0, 1] - height / 2) < 1, "Point should be at vertical center"


def test_project_point_to_the_right():
    """Test projecting a point to the right of the camera."""
    # Camera at origin looking along +x axis
    tf_camera_to_episodic = xyz_yaw_to_tf_matrix(np.array([0, 0, 1.5]), 0)

    # Point 5 meters in front, 2 meters to the right
    map_points = np.array([[5.0, -2.0]])

    # Camera parameters
    fx, fy = 320.0, 320.0
    width, height = 640, 480

    pixel_coords, valid_mask = project_map_points_to_image(
        map_points_xy=map_points,
        tf_camera_to_episodic=tf_camera_to_episodic,
        fx=fx,
        fy=fy,
        image_width=width,
        image_height=height,
        point_height=1.5,
    )

    # Point should be valid and to the right of center
    assert valid_mask[0], "Point should be visible"
    assert pixel_coords[0, 0] > width / 2, "Point should be to the right of center"
    assert np.abs(pixel_coords[0, 1] - height / 2) < 1, "Point should be at vertical center"


def test_project_point_behind_camera():
    """Test that points behind the camera are marked as invalid."""
    # Camera at origin looking along +x axis
    tf_camera_to_episodic = xyz_yaw_to_tf_matrix(np.array([0, 0, 1.5]), 0)

    # Point behind the camera
    map_points = np.array([[-5.0, 0.0]])

    fx, fy = 320.0, 320.0
    width, height = 640, 480

    pixel_coords, valid_mask = project_map_points_to_image(
        map_points_xy=map_points,
        tf_camera_to_episodic=tf_camera_to_episodic,
        fx=fx,
        fy=fy,
        image_width=width,
        image_height=height,
        point_height=1.5,
    )

    # Point should be invalid
    assert not valid_mask[0], "Point behind camera should not be visible"


def test_project_multiple_points():
    """Test projecting multiple frontiers at once."""
    # Camera at origin looking along +x axis
    tf_camera_to_episodic = xyz_yaw_to_tf_matrix(np.array([0, 0, 1.5]), 0)

    # Multiple points: front, right, left, behind
    map_points = np.array(
        [
            [5.0, 0.0],  # front
            [5.0, -2.0],  # front-right
            [5.0, 2.0],  # front-left
            [-5.0, 0.0],  # behind
        ]
    )

    fx, fy = 320.0, 320.0
    width, height = 640, 480

    pixel_coords, valid_mask = project_map_points_to_image(
        map_points_xy=map_points,
        tf_camera_to_episodic=tf_camera_to_episodic,
        fx=fx,
        fy=fy,
        image_width=width,
        image_height=height,
        point_height=1.5,
    )

    # First three should be valid, last one invalid
    assert valid_mask[0], "Front point should be visible"
    assert valid_mask[1], "Front-right point should be visible"
    assert valid_mask[2], "Front-left point should be visible"
    assert not valid_mask[3], "Behind point should not be visible"

    # Check left-right ordering
    assert pixel_coords[1, 0] > pixel_coords[0, 0], "Right point should have larger u"
    assert pixel_coords[2, 0] < pixel_coords[0, 0], "Left point should have smaller u"


def test_empty_points_array():
    """Test handling of empty points array."""
    tf_camera_to_episodic = xyz_yaw_to_tf_matrix(np.array([0, 0, 1.5]), 0)
    map_points = np.array([]).reshape(0, 2)

    fx, fy = 320.0, 320.0
    width, height = 640, 480

    pixel_coords, valid_mask = project_map_points_to_image(
        map_points_xy=map_points,
        tf_camera_to_episodic=tf_camera_to_episodic,
        fx=fx,
        fy=fy,
        image_width=width,
        image_height=height,
        point_height=1.5,
    )

    assert pixel_coords.shape == (0, 2), "Should return empty array"
    assert valid_mask.shape == (0,), "Should return empty mask"


def test_rotated_camera():
    """Test projection with a rotated camera."""
    # Camera looking at 90 degrees (along +y axis)
    tf_camera_to_episodic = xyz_yaw_to_tf_matrix(np.array([0, 0, 1.5]), np.pi / 2)

    # Point at (0, 5) - directly in front of rotated camera
    map_points = np.array([[0.0, 5.0]])

    fx, fy = 320.0, 320.0
    width, height = 640, 480

    pixel_coords, valid_mask = project_map_points_to_image(
        map_points_xy=map_points,
        tf_camera_to_episodic=tf_camera_to_episodic,
        fx=fx,
        fy=fy,
        image_width=width,
        image_height=height,
        point_height=1.5,
    )

    assert valid_mask[0], "Point should be visible"
    assert np.abs(pixel_coords[0, 0] - width / 2) < 1, "Point should be at horizontal center"
    assert np.abs(pixel_coords[0, 1] - height / 2) < 1, "Point should be at vertical center"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

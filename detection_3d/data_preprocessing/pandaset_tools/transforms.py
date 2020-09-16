#!/usr/bin/env python
__copyright__ = """
Copyright (c) 2020 Tananaev Denis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this permission
notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import numpy as np


def quaternion_to_euler(w, x, y, z):
    """
    Converts quaternions with components w, x, y, z into a tuple (roll, pitch, yaw)
    
    """
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x ** 2 + y ** 2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y ** 2 + z ** 2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):

    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
            [np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def intrinsics_to_matrix(intrinsics):
    intrinsics_matrix = np.eye(3)
    intrinsics_matrix[0, 0] = intrinsics["fx"]
    intrinsics_matrix[0, 2] = intrinsics["cx"]
    intrinsics_matrix[1, 1] = intrinsics["fy"]
    intrinsics_matrix[1, 2] = intrinsics["cy"]
    return intrinsics_matrix


def to_transform_matrix(translation, rotation):
    Rt = np.eye(4)
    Rt[:3, :3] = eulerAnglesToRotationMatrix(rotation)
    Rt[:3, 3] = translation
    return Rt


def transform_lidar_to_camera(lidar, Rt):

    lidar_3d = lidar[:, :3]
    lidar_3d = np.transpose(lidar_3d)

    ones = np.ones_like(lidar_3d[0])[None, :]
    hom_coord = np.concatenate((lidar_3d, ones), axis=0)
    lidar_3d = np.dot(Rt, hom_coord)
    lidar_3d = np.transpose(lidar_3d)[:, :3]

    return lidar_3d


def transform_project_lidar_to_image(lidar_3d, intrinsics, max_x=1920, max_y=1080):
    """
    Arguments:
     lidar_3d: (num_points, 3) lidar array in camera coordinates
     intrinsics: (3, 3) camera intrinsics
    """
    lidar_3d = np.transpose(lidar_3d)
    pixels_2d = np.matmul(intrinsics, lidar_3d)
    pixels_2d = pixels_2d / pixels_2d[-1]

    lidar_3d = np.transpose(lidar_3d)
    pixels_2d = np.transpose(pixels_2d)

    fov_inds = (
        (pixels_2d[:, 0] < max_x)
        & (pixels_2d[:, 0] >= 0.0)
        & (pixels_2d[:, 1] < max_y)
        & (pixels_2d[:, 1] >= 0.0)
    )
    fov_inds = fov_inds & (lidar_3d[:, -1] > 0.0)

    pixels_2d = pixels_2d[fov_inds]
    lidar_3d = lidar_3d[fov_inds]

    return pixels_2d[:, :-1], lidar_3d

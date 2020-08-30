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
import argparse
import numpy as np
import os
import glob
from PIL import Image
import pandas as pd
from detection_3d.tools.file_io import read_json, save_depth_16bit
from detection_3d.data_preprocessing.pandaset_tools.helpers import (
    make_depth,
    make_point_cloud,
    make_xzyhwly,
    filter_boxes,
)
from detection_3d.data_preprocessing.pandaset_tools.transforms import (
    quaternion_to_euler,
    to_transform_matrix,
    transform_lidar_to_camera,
    intrinsics_to_matrix,
    transform_project_lidar_to_image,
)
from tqdm import tqdm
from detection_3d.tools.visualization_tools import (
    visualize_lidar,
    visualize_bboxes_3d,
    visualize_colored_pc,
)
import mayavi.mlab as mlab

from detection_3d.tools.detection_helpers import (
    make_eight_points_boxes,
    get_bboxes_parameters_from_points,
)


def get_transformation_matrix(pose, idx):
    # Get pose of the lidar
    translation = pose[idx]["position"]
    translation = np.asarray([translation[key] for key in translation])
    rotation = pose[idx]["heading"]
    rotation = np.asarray([rotation[key] for key in rotation])
    rotation = quaternion_to_euler(*rotation)
    Rt = to_transform_matrix(translation, rotation)
    return Rt


def process_data(dataset_dir, camera="front_camera"):
    """
    The function visualizes data from pandaset.
    Arguments:
        dataset_dir: directory with  Pandaset data
    """

    # Get list of data samples
    search_string = os.path.join(dataset_dir, "*")
    seq_list = sorted(glob.glob(search_string))
    for seq in tqdm(seq_list, desc="Process sequences", total=len(seq_list)):
        # Make output dirs for data
        depth_dir = os.path.join(seq, "vidar", camera, "depth")  # depth images
        boxes_dir = os.path.join(
            seq, "vidar", camera, "boxes"
        )  # boxes in camera coordinates
        # get lidar
        search_string = os.path.join(seq, "lidar", "*.pkl.gz")
        lidar_list = sorted(glob.glob(search_string))
        lidar_pose_path = os.path.join(seq, "lidar", "poses.json")
        lidar_pose = read_json(lidar_pose_path)
        # get camera pose
        camera_pose_path = os.path.join(seq, "camera", camera, "poses.json")
        camera_pose = read_json(camera_pose_path)
        camera_intrinsics_path = os.path.join(seq, "camera", camera, "intrinsics.json")
        intrinsics = read_json(camera_intrinsics_path)
        intr = intrinsics_to_matrix(intrinsics)

        for idx, lidar_path in enumerate(lidar_list):
            sample_idx = os.path.splitext(os.path.basename(lidar_path))[0].split(".")[0]
            world_T_camera = get_transformation_matrix(camera_pose, idx)
            camera_T_world = np.linalg.inv(world_T_camera)
            # camera_T_lidar = np.dot(camera_T_world, world_T_lidar)
            # Get respective image
            image_path = lidar_path.split("/")
            image_path[-2] = "camera/" + camera
            image_path[-1] = image_path[-1].split(".")[0] + ".jpg"
            image_path = os.path.join(*image_path)

            # Get respective bboxes
            bbox_path = lidar_path.split("/")
            bbox_path[-2] = "annotations/cuboids"
            bbox_path = os.path.join(*bbox_path)
            # Load data
            lidar = np.asarray(pd.read_pickle(lidar_path))
            lidar = lidar[lidar[:, -1] == 1]

            image = np.asarray(Image.open(image_path))

            # Load bboxes
            bboxes = np.asarray(pd.read_pickle(bbox_path))
            labels, bboxes = make_xzyhwly(bboxes)
            corners_3d, orientation_3d = make_eight_points_boxes(bboxes)
            corners_3d = np.asarray(
                [transform_lidar_to_camera(box, camera_T_world) for box in corners_3d]
            )

            lidar = transform_lidar_to_camera(lidar, camera_T_world)
            points_2d, lidar_fov = transform_project_lidar_to_image(lidar, intr)
            depth_image = make_depth(lidar_fov, points_2d)
            depth_image[depth_image > 80] = 0.0

            pc, color = make_point_cloud(image, depth_image, intr)
            labels, corners_3d = filter_boxes(labels, corners_3d, pc)

            figure = visualize_bboxes_3d(corners_3d, None, None)
            figure = visualize_lidar(pc, figure=figure)

            depth_image = depth_image * 100
            save_depth_16bit("depth.png", depth_image)
            print(f"depth_image min {np.min(depth_image)} max {np.max(depth_image)}")

            mlab.show(1)
            input()
            mlab.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess 3D pandaset.")
    parser.add_argument("--dataset_dir", default="../../dataset")
    args = parser.parse_args()
    process_data(args.dataset_dir)

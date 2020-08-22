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
from detection_3d.tools.file_io import read_json
from tqdm import tqdm


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
    print(dataset_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess 3D pandaset.")
    parser.add_argument("--dataset_dir", default="../../dataset")
    args = parser.parse_args()
    process_data(args.dataset_dir)

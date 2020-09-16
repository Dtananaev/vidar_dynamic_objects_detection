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
import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm
from detection_3d.parameters import Parameters
from detection_3d.tools.file_io import (
    load_dataset_list,
    load_image,
    load_bboxes,
    load_depth_16bit,
    read_json,
)
from PIL import Image


class VidarDataset:
    """
    This is dataset layer for 3d detection experiment
    Arguments:
        param_settings: parameters of experiment
        dataset_file: name of .dataset file
        shuffle: shuffle the data True/False
    """

    def __init__(self, param_settings, dataset_file, augmentation=False, shuffle=False):
        # Private methods
        self.seed = param_settings["seed"]
        np.random.seed(self.seed)

        self.augmentation = augmentation

        self.param_settings = param_settings
        self.dataset_file = dataset_file
        self.inputs_list = load_dataset_list(
            self.param_settings["dataset_dir"], dataset_file
        )
        self.num_samples = len(self.inputs_list)
        self.num_it_per_epoch = int(
            self.num_samples / self.param_settings["batch_size"]
        )
        self.output_types = [tf.float32, tf.float32]

        ds = tf.data.Dataset.from_tensor_slices(self.inputs_list)

        if shuffle:
            ds = ds.shuffle(self.num_samples)
        ds = ds.map(
            map_func=lambda x: tf.py_function(
                self.load_data, [x], Tout=self.output_types
            ),
            num_parallel_calls=12,
        )
        ds = ds.batch(self.param_settings["batch_size"])
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.dataset = ds

    def load_data(self, data_input):
        """
        Loads image, depth, boxes, intrinsics
        """
        image_file, depth_file, boxes_file, intr_file = np.asarray(data_input).astype(
            "U"
        )

        image = load_image(image_file)
        image = image[:1024, :, :]
        image /= 255.0
        depth = load_depth_16bit(depth_file)
        depth = depth[:1024, :]

        depth = tf.cast(depth, tf.float32)  # in cm
        depth /= 100.0  # in meters

        depth = tf.where(depth == 0.0, np.nan, depth)
        disparity = 1.0 / depth
        disparity = tf.expand_dims(disparity, axis=-1)

        return image, disparity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DatasetLayer.")
    parser.add_argument(
        "--dataset_file",
        type=str,
        help="creates .dataset file",
        default="train.datatxt",
    )
    args = parser.parse_args()

    param_settings = Parameters().settings
    train_dataset = VidarDataset(param_settings, args.dataset_file)

    for samples in tqdm(train_dataset.dataset, total=train_dataset.num_it_per_epoch):
        images, disparity = samples
        print(f"images {images.shape}, disparity {disparity.shape}")
        # disparity = tf.where(tf.math.is_finite(disparity), disparity, 0.0)
        # disparity = disparity.numpy()

        # print(f"disparity {np.min(disparity)}, max {np.max(disparity)}")
        img = images[0].numpy() * 255
        disp = disparity[0].numpy()
        # depth = 1.0 / disp * 100
        disp = np.where(np.isnan(disp), 0.0, disp)
        # print(f"depth {depth.shape}")
        disp = 255 * disp / np.max(disp)
        img = Image.fromarray(img.astype("uint8"))
        depth = Image.fromarray(disp[:, :, 0].astype("uint8"))
        depth.save("depth.png")
        img.save("img.png")
        input()

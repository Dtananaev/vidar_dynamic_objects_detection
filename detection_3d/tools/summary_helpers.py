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
from detection_3d.tools.helpers import normalize, replace_nonfinite


def train_summaries(train_out, optimizer, param_settings, learning_rate):
    """
    Visualizes  the train outputs in tensorboards
    """

    writer = tf.summary.create_file_writer(param_settings["train_summaries"])
    with writer.as_default():
        # Losses
        total_depth_loss = train_out["total_depth_loss"]

        # Show learning rate given scheduler
        if param_settings["scheduler"]["name"] != "no_scheduler":
            with tf.name_scope("Optimizer info"):
                step = float(
                    optimizer.iterations.numpy()
                )  # triangular_scheduler learning rate needs float dtype
                tf.summary.scalar(
                    "learning_rate", learning_rate(step), step=optimizer.iterations
                )
        with tf.name_scope("Training losses"):
            tf.summary.scalar(
                "1.Total loss", train_out["total_loss"], step=optimizer.iterations
            )
            tf.summary.scalar(
                "2. total_depth_loss", total_depth_loss, step=optimizer.iterations
            )

        if (
            param_settings["step_summaries"] is not None
            and optimizer.iterations % param_settings["step_summaries"] == 0
        ):
            with tf.name_scope("0-Input"):
                tf.summary.image(
                    "Images", train_out["images"], step=optimizer.iterations
                )

            # Show GT
            with tf.name_scope("1-Ground truth disparity"):
                gt_disp = replace_nonfinite(train_out["gt_disp"])
                tf.summary.image("Gt", normalize(gt_disp), step=optimizer.iterations)

            with tf.name_scope("2-Predicted disparity"):
                tf.summary.image(
                    "Prediction",
                    normalize(train_out["disp"]),
                    step=optimizer.iterations,
                )


# def epoch_metrics_summaries(param_settings, epoch_metrics, epoch):
#     """
#     Visualizes epoch metrics
#     """
#     # Train results
#     writer = tf.summary.create_file_writer(param_settings["train_summaries"])
#     with writer.as_default():
#         # Show epoch metrics for train
#         with tf.name_scope("Epoch metrics"):
#             tf.summary.scalar(
#                 "1. Loss", epoch_metrics.train_loss.result().numpy(), step=epoch
#             )

#     # Val results
#     writer = tf.summary.create_file_writer(param_settings["eval_summaries"])
#     with writer.as_default():
#         # Show epoch metrics for train
#         with tf.name_scope("Epoch metrics"):
#             tf.summary.scalar(
#                 "1. Loss", epoch_metrics.val_loss.result().numpy(), step=epoch
#             )

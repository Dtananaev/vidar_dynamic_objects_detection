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
import mayavi.mlab as mlab
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import pptk

# from detection_3d.tools.detection_helpers import (
#     get_boxes_from_box_grid,
#     make_eight_points_boxes,
# )
from detection_3d.data_preprocessing.pandaset_tools.helpers import get_color


def visualize_colored_pc(point_cloud, color):
    v = pptk.viewer(point_cloud)
    v.attributes(color / 255.0)
    v.set(point_size=0.005)


def visualize_lidar(lidar, color=None, figure=None):
    """ 
    Draw lidar points
    Args:
        lidar: numpy array (n,3) of XYZ
        figure: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    """

    if figure is None:
        figure = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
        )

    if color is None:
        color = lidar[:, 2]
    mlab.points3d(
        lidar[:, 0],
        lidar[:, 1],
        lidar[:, 2],
        color,
        mode="point",
        scale_factor=0.3,
        figure=figure,
    )

    # draw origin
    mlab.points3d(
        0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2, figure=figure
    )
    # draw axis
    mlab.plot3d(
        [0, 2], [0, 0], [0, 0], color=(1, 0, 0), tube_radius=None, figure=figure
    )
    mlab.plot3d(
        [0, 0], [0, 2], [0, 0], color=(0, 1, 0), tube_radius=None, figure=figure
    )
    mlab.plot3d(
        [0, 0], [0, 0], [0, 2], color=(0, 0, 1), tube_radius=None, figure=figure
    )
    return figure


def visualize_bboxes_3d(lidar_corners_3d, figure=None, orientation=None):
    if figure is None:
        figure = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
        )

    for b in tqdm(lidar_corners_3d, desc=f"Add bboxes", total=len(lidar_corners_3d)):
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=(1, 1, 1),
                tube_radius=None,
                line_width=1,
                figure=figure,
            )

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=(1, 1, 1),
                tube_radius=None,
                line_width=1,
                figure=figure,
            )

            i, j = k, k + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=(1, 1, 1),
                tube_radius=None,
                line_width=1,
                figure=figure,
            )
    if orientation is not None:
        for o in orientation:
            mlab.plot3d(
                [o[0, 0], o[1, 0]],
                [o[0, 1], o[1, 1]],
                [o[0, 2], o[1, 2]],
                color=(1, 1, 1),
                tube_radius=None,
                line_width=1,
                figure=figure,
            )
    print(f"Done")
    return figure


def draw_boxes_top_view(
    top_view_image, boxes_3d, grid_meters, labels, orientation_3d=None
):
    height, width, channels = top_view_image.shape
    delimiter_x = grid_meters[0] / height
    delimiter_y = grid_meters[1] / width
    thickness = 2
    for idx, b in enumerate(boxes_3d):
        color = get_color(labels[idx]) / 255
        b = b[:4]
        x = np.floor(b[:, 0] / delimiter_x).astype(int)

        y = np.floor(b[:, 1] / delimiter_y).astype(int)

        cv2.line(top_view_image, (y[0], x[0]), (y[1], x[1]), color, thickness)
        cv2.line(top_view_image, (y[1], x[1]), (y[2], x[2]), color, thickness)
        cv2.line(top_view_image, (y[2], x[2]), (y[3], x[3]), color, thickness)
        cv2.line(top_view_image, (y[3], x[3]), (y[0], x[0]), color, thickness)

    if orientation_3d is not None:
        for o in orientation_3d:
            x = np.floor(o[:, 0] / delimiter_x).astype(int)
            y = np.floor(o[:, 1] / delimiter_y).astype(int)
            cv2.arrowedLine(
                top_view_image, (y[0], x[0]), (y[1], x[1]), (1, 0, 0), thickness
            )
    return top_view_image


# def visualize_2d_boxes_on_top_image(
#     bboxes_grid, top_view, grid_meters, bbox_voxel_size, prediction=False
# ):
#     top_image_vis = []
#     for boxes, top_image in zip(bboxes_grid, top_view):  # iterate over batch
#         top_image = top_image.numpy()
#         shape = top_image.shape
#         rgb_image = np.zeros((shape[0], shape[1], 3))
#         rgb_image[top_image[:, :, 0] > 0] = 1

#         box, labels, _ = get_boxes_from_box_grid(boxes, bbox_voxel_size)
#         box = box.numpy()
#         box, orientation_3d = make_eight_points_boxes(box)

#         if prediction:
#             labels = np.argmax(labels, axis=-1)
#         if len(box) > 0:
#             rgb_image = draw_boxes_top_view(
#                 rgb_image, box, grid_meters, labels, orientation_3d
#             )

#         # rgb_image = np.rot90(rgb_image)
#         top_image_vis.append(rgb_image)
#     return np.asarray(top_image_vis)


def visualize_bboxes_on_image(image, bboxes_2d, labels, orientation_2d=None):
    """
    The function visualize the reprojected 3d bounding boxes
    on 2d image
    Arguments:
         images: the tensor of the shape [height, width, 3]
         bboxes: the reprojected bboxes of the shape [num_boxes, 8, 2]
    Returns:
         resulted_images: the tensor with bboxes of the shape [height, width, 3]
    """

    height, width, _ = image.shape
    thickness = 2
    boundaries = np.asarray([width, height])
    for idx, b in enumerate(bboxes_2d):
        color = get_color(labels[idx]) / 255

        b = b.astype(np.int32)
        first_square = False
        second_square = False
        if (
            (b[0] >= 0).all() & (b[0] < boundaries).all()
            or (b[1] >= 0).all() & (b[1] < boundaries).all()
            or (b[4] >= 0).all() & (b[4] < boundaries).all()
            or (b[5] >= 0).all() & (b[5] < boundaries).all()
        ):
            first_square = True
            cv2.line(image, (b[0, 0], b[0, 1]), (b[1, 0], b[1, 1]), color, thickness)
            cv2.line(image, (b[4, 0], b[4, 1]), (b[0, 0], b[0, 1]), color, thickness)
            cv2.line(image, (b[5, 0], b[5, 1]), (b[1, 0], b[1, 1]), color, thickness)
            cv2.line(image, (b[4, 0], b[4, 1]), (b[5, 0], b[5, 1]), color, thickness)
        if (
            (b[2] >= 0).all() & (b[2] < boundaries).all()
            or (b[3] >= 0).all() & (b[3] < boundaries).all()
            or (b[6] >= 0).all() & (b[6] < boundaries).all()
            or (b[7] >= 0).all() & (b[7] < boundaries).all()
        ):
            second_square = True
            cv2.line(image, (b[2, 0], b[2, 1]), (b[3, 0], b[3, 1]), color, thickness)
            cv2.line(image, (b[6, 0], b[6, 1]), (b[2, 0], b[2, 1]), color, thickness)
            cv2.line(image, (b[7, 0], b[7, 1]), (b[3, 0], b[3, 1]), color, thickness)
            cv2.line(image, (b[7, 0], b[7, 1]), (b[6, 0], b[6, 1]), color, thickness)

        if first_square and second_square:
            cv2.line(image, (b[0, 0], b[0, 1]), (b[3, 0], b[3, 1]), color, thickness)
            cv2.line(image, (b[1, 0], b[1, 1]), (b[2, 0], b[2, 1]), color, thickness)
            cv2.line(image, (b[4, 0], b[4, 1]), (b[7, 0], b[7, 1]), color, thickness)
            cv2.line(image, (b[5, 0], b[5, 1]), (b[6, 0], b[6, 1]), color, thickness)

    if orientation_2d is not None:
        for o in orientation_2d:
            o = o.astype(np.int32)
            if (
                (o[0] >= 0).all()
                & (o[0] < boundaries).all()
                & (o[1] >= 0).all()
                & (o[1] < boundaries).all()
            ):
                cv2.arrowedLine(
                    image, (o[0, 0], o[0, 1]), (o[1, 0], o[1, 1]), (1, 0, 0), thickness
                )

    return image

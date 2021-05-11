# Copyright (C) 2017-2020 Trent Houliston <trent@houliston.me>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import hashlib
import math
import os

import cv2
import numpy as np

import tensorflow as tf
from training.op import difference_visual_mesh, map_visual_mesh, unmap_visual_mesh
from training.projection import project


class SeekerImages(tf.keras.callbacks.Callback):
    def __init__(self, output_path, dataset, model, max_distance, geometry, radius, scale):
        super(SeekerImages, self).__init__()

        self.max_distance = max_distance
        self.radius = radius
        self.map_args = {"model": model, "geometry": geometry, "radius": radius}
        self.scale = scale
        self.writer = tf.summary.create_file_writer(os.path.join(output_path, "images"))

        # Load the dataset and extract a single record from it
        for d in dataset:
            data = d

        self.X = data["X"]
        self.Y = data["Y"]
        self.G = data["G"]
        self.Hoc = tf.reshape(data["Hoc"], (-1, 4, 4))
        self.img = tf.reshape(data["jpg"], (-1,))
        self.lens = {
            "projection": tf.reshape(data["lens"]["projection"], (-1,)),
            "focal_length": tf.reshape(data["lens"]["focal_length"], (-1,)),
            "centre": tf.reshape(data["lens"]["centre"], (-1, 2)),
            "k": tf.reshape(data["lens"]["k"], (-1, 2)),
            "fov": tf.reshape(data["lens"]["fov"], (-1,)),
        }

        # Work out the data ranges
        cs = [0] + np.cumsum(data["n"]).tolist()
        self.ranges = list(zip(cs, cs[1:]))
        self.nm = tf.concat(
            [
                unmap_visual_mesh(data["V"][r[0] : r[1]], height=self.Hoc[i][2, 3], **self.map_args)
                for i, r in enumerate(self.ranges)
            ],
            axis=0,
        )

    def _on_screen(self, px, dims):

        # Return the indices of px that are within the bounds of dimensions
        return tf.squeeze(
            tf.where(tf.reduce_all(tf.logical_and(px >= 0, px <= tf.expand_dims(dims, 0)), axis=-1)), axis=-1
        )

    def _ring(self, angle, width, f):

        # This is an ok approximation of how much angle makes a pixel
        # r = f * theta is true equidistant lens so it works here
        # In rectilinear lenses we just assume that theta = tan(theta) is an ok approximation
        px_angle = tf.math.reciprocal(f)

        # An estimate of how many points will be around the ring (x2 to be sure we hit all the pixels)
        pts = 2 * tf.cast(2.0 * math.pi * f * tf.math.tan(angle), dtype=tf.int32)

        # Our Z value in is based on our cone angle ± a half width
        # Here, Z is a negative value, as we are looking towards the plane (Spotlight)
        theta = tf.linspace(angle - 0.5 * width * px_angle, angle + 0.5 * width * px_angle, 2 * width)
        z = -tf.cos(theta)
        w = tf.expand_dims(tf.sin(theta), 1)

        # Make a circle in x and y and weight them by -sin(theta) to make the unit vector ring
        # Then adjust the length by sin(theta) to make the circle a unit vector ring
        psi = tf.linspace(0.0, 2.0 * math.pi, pts)
        x = tf.expand_dims(tf.cos(psi), 0) * w
        y = tf.expand_dims(tf.sin(psi), 0) * w

        return tf.reshape(tf.stack([x, y, tf.broadcast_to(tf.expand_dims(z, axis=1), x.shape)], axis=-1), [-1, 3])

    def _bees(self, X, Y, px, dims):

        on_screen = self._on_screen(px, dims)

        # Gather the on screen points
        X = tf.gather(X, on_screen)
        Y = tf.gather(Y, on_screen)
        px = tf.gather(px, on_screen)

        # Work out if X or Y is close enough that this is, or should be a prediction
        X_near = tf.reduce_all(tf.math.abs(X) <= 0.75, axis=-1)
        Y_near = tf.reduce_all(tf.math.abs(Y) <= 0.75, axis=-1)

        # Work out our various states so we can draw dots for them
        tp = tf.squeeze(tf.where(tf.logical_and(X_near, Y_near)), axis=-1)
        fp = tf.squeeze(tf.where(tf.logical_and(X_near, tf.logical_not(Y_near))), axis=-1)
        fn = tf.squeeze(tf.where(tf.logical_and(tf.logical_not(X_near), Y_near)), axis=-1)
        tn = tf.squeeze(tf.where(tf.logical_and(tf.logical_not(X_near), tf.logical_not(Y_near))), axis=-1)

        # Weight based on how close it is to the point
        weight = tf.clip_by_value(1.0 - (tf.norm(X, axis=-1) / math.sqrt(2.0 * 0.75 ** 2.0)), 0.0, 1.0)

        # Gather the weights into the image grid
        tp = tf.scatter_nd(tf.gather(px, tp), tf.gather(weight, tp), dims)
        fp = tf.scatter_nd(tf.gather(px, fp), tf.gather(weight, fp), dims)
        fn = tf.scatter_nd(tf.gather(px, fn), tf.gather(weight, fn), dims)

        # Apply the colours
        tp = tf.einsum("ij,k->ijk", tp, tf.constant([1.0, 1.0, 0.0]))
        fp = tf.einsum("ij,k->ijk", fp, tf.constant([1.0, 0.0, 0.0]))
        fn = tf.einsum("ij,k->ijk", fn, tf.constant([0.0, 0.0, 1.0]))

        # Merge into an overlay
        return tf.clip_by_value(tp + fp + fn, 0, 1.0)

    def _blend(self, a, b):
        return a * (1.0 - tf.reduce_max(b, axis=-1, keepdims=True)) + b

    def image(self, img, X, Y, Hoc, lens, nm):

        # Y true contains a list of all the possible answers, we are ranked on how close we are to the closest one
        # This code works out the closest point to our prediction for each case so we can use that for training
        Y = tf.gather_nd(
            Y,
            tf.stack(
                [
                    tf.range(tf.shape(Y)[0], dtype=tf.int32),
                    tf.math.argmin(
                        tf.reduce_sum(tf.math.squared_difference(tf.expand_dims(X, axis=-2), Y), axis=-1),
                        axis=-1,
                        output_type=tf.int32,
                    ),
                ],
                axis=1,
            ),
        )

        # hash of the image file for sorting later
        img_hash = hashlib.md5()
        img_hash.update(img)
        img_hash = img_hash.digest()

        # Decode the image and convert it to float32
        img = tf.image.convert_image_dtype(tf.image.decode_image(img, channels=3, expand_animations=False), tf.float32)

        # Extract lens parameters
        projection = lens["projection"]
        f = lens["focal_length"]
        centre = lens["centre"]
        k = lens["k"]
        dims = img.shape[:2]

        # Convert the predictions into unit vectors in camera space and project them to pixel coordinates
        uTCo = map_visual_mesh(nm + X * self.scale, height=Hoc[2, 3], **self.map_args)
        uTCc = tf.einsum("ij,ki->kj", Hoc[:3, :3], uTCo)
        target_px = tf.cast(tf.round(project(uTCc, dims, projection, f, centre, k)), tf.int32)

        # Draw an overlay of predictions
        bees = self._bees(X, Y, target_px, dims)

        # Get pixel coordinates for a ring around the search region
        uRCo = self._ring(angle=tf.atan(self.max_distance / Hoc[2, 3]), width=2, f=f)
        uRCc = tf.einsum("ij,ki->kj", Hoc[:3, :3], uRCo)
        ring_px = tf.cast(tf.round(project(uRCc, dims, projection, f, centre, k)), tf.int32)

        # Get the on screen pixels
        ring_px = tf.gather(ring_px, self._on_screen(ring_px, dims))
        ring_overlay = tf.scatter_nd(ring_px, tf.ones_like(ring_px[:, 0], dtype=tf.float32), dims)

        # Apply the colours
        ring_overlay = tf.clip_by_value(tf.einsum("ij,k->ijk", ring_overlay, tf.constant([1.0, 1.0, 1.0])), 0.0, 1.0)

        output = self._blend(self._blend(img, ring_overlay), bees)

        return (img_hash, output)

    def on_epoch_end(self, epoch, logs=None):

        # Make a dataset that we can infer from, we need to make the input a tuple in a tuple.
        # If it is not it considers G to be Y and it fails to execute
        # Then using this dataset of images, do a prediction using the model
        predictions = self.model((self.X, self.G))

        # Work out the valid data ranges for each of the objects
        images = []
        for i, r in enumerate(self.ranges):
            images.append(
                self.image(
                    img=self.img[i].numpy(),
                    X=predictions[r[0] : r[1]],
                    Y=self.Y[r[0] : r[1]],
                    Hoc=self.Hoc[i],
                    lens={k: v[i] for k, v in self.lens.items()},
                    nm=self.nm[r[0] : r[1]],
                )
            )

        # Write out the images to tensorboard
        # Sort by hash so the images show up in the same order every time
        with self.writer.as_default():
            for i, img in enumerate(sorted(images, key=lambda image: image[0])):
                tf.summary.image("images/{}".format(i), tf.expand_dims(img[1], axis=0), step=epoch, max_outputs=1)

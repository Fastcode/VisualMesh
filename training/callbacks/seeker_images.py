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
import io
import os
import warnings

import cv2
import numpy as np

from training.projection import project
import tensorflow as tf
from training.op import difference_visual_mesh, map_visual_mesh, unmap_visual_mesh


class SeekerImages(tf.keras.callbacks.Callback):
    def __init__(self, output_path, dataset, model, geometry, radius, scale):
        super(SeekerImages, self).__init__()

        self.map_args = {"model": model, "geometry": geometry, "radius": radius}
        self.scale = scale
        self.writer = tf.summary.create_file_writer(os.path.join(output_path, "images"))

        # Load the dataset and extract a single record from it
        for d in dataset:
            data = d

        self.X = data["X"]
        self.G = data["G"]
        self.C = data["C"]
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

    def image(self, img, C, X, Hoc, lens, nm):

        # hash of the image file for sorting later
        img_hash = hashlib.md5()
        img_hash.update(img)
        img_hash = img_hash.digest()

        # Decode the image
        img = cv2.cvtColor(cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # Convert the predictions into unit vectors in camera space
        rTCo = map_visual_mesh(nm + X * self.scale, height=Hoc[2, 3], **self.map_args)
        rTCc = tf.einsum("ij,ki->kj", Hoc[:3, :3], rTCo)

        target = project(rTCc, img.shape[:2], lens["projection"], lens["focal_length"], lens["centre"], lens["k"])

        for origin, target in zip(C, target):
            img = cv2.arrowedLine(
                img, tuple(reversed(origin.numpy())), tuple(reversed(target.numpy())), (255, 255, 255)
            )

        return (img_hash, tf.convert_to_tensor(img))

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
                    C=self.C[r[0] : r[1]],
                    X=predictions[r[0] : r[1]],
                    Hoc=self.Hoc[i],
                    lens={k: v[i] for k, v in self.lens.items()},
                    nm=self.nm[r[0] : r[1]],
                )
            )

        # Sort by hash so the images show up in the same order every time
        images = tf.stack([i for h, i in sorted(images)], axis=0)

        with self.writer.as_default():
            # Write the images
            tf.summary.image("images", images, step=epoch, max_outputs=images.shape[0])

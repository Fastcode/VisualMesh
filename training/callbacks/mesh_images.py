#!/usr/bin/env python3

import hashlib
import io
import os
import warnings

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from training.dataset import VisualMeshDataset

mpl.use("Agg")


class MeshImages(tf.keras.callbacks.Callback):
    def __init__(self, output_path, dataset_path, classes, model, progress_images, colours):

        self.colours = colours
        self.writer = tf.summary.create_file_writer(os.path.join(output_path, "images"))

        # Load the dataset and extract a single record from it
        for d in (
            VisualMeshDataset(
                input_files=dataset_path,
                classes=classes,
                model=model,
                batch_size=progress_images,
                prefetch=tf.data.experimental.AUTOTUNE,
                variants={},
            )
            .build()
            .take(1)
        ):
            self.data = d

    def mesh_image(self, img, px, X, colours):

        # hash of the image file for sorting later
        img_hash = hashlib.md5()
        img_hash.update(img)
        img_hash = img_hash.digest()

        # Decode the image
        img = cv2.cvtColor(cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # Setup the display so everything is all at the correct resolution
        dpi = 80
        height, width, channels = img.shape
        figsize = width / float(dpi), height / float(dpi)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        # Image underlay
        ax.imshow(img, interpolation="nearest")

        # We need at least 3 points to make a triangle
        if px.shape[0] >= 3:

            # Stop matplotlib complaining
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                for i, colour in enumerate(colours):
                    colour = np.array(colour) / 255
                    ax.tricontour(
                        px[:, 1],
                        px[:, 0],
                        X[:, i],
                        levels=[0.5, 0.75, 0.9],
                        colors=[(*colour, 0.33), (*colour, 0.66), (*colour, 1.0)],
                    )

        ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

        # Write the image as a jpg to a BytesIO and return it
        data = io.BytesIO()
        fig.savefig(data, format="raw", dpi=dpi)
        ax.cla()
        fig.clf()
        plt.close(fig)
        data.seek(0)

        # Convert the image from raw format into a height*width*3
        data = tf.reshape(tf.io.decode_raw(data.read(), tf.uint8), (height, width, 4))[:, :, :3]

        return (img_hash, data)

    def on_epoch_end(self, epoch, logs=None):

        # Make a dataset that we can infer from, we need to make the input a tuple in a tuple.
        # If it is not it considers G to be Y and it fails to execute
        # Then using this dataset of images, do a prediction using the model
        ds = tf.data.Dataset.zip(
            ((tf.data.Dataset.from_tensors(self.data["X"]), tf.data.Dataset.from_tensors(self.data["G"]),),)
        )
        predictions = self.model.predict(ds)

        # Work out the valid data ranges for each of the objects
        cs = [0] + np.cumsum(self.data["n"]).tolist()
        ranges = list(zip(cs, cs[1:]))
        images = []
        for i, r in enumerate(ranges):
            images.append(
                self.mesh_image(
                    self.data["raw"][i].numpy(), self.data["px"][r[0] : r[1]], predictions[r[0] : r[1]], self.colours,
                )
            )

        # Sort by hash so the images show up in the same order every time
        images = tf.stack([i for h, i in sorted(images)], axis=0)

        with self.writer.as_default():
            # Write the images
            tf.summary.image("images", images, step=epoch, max_outputs=images.shape[0])

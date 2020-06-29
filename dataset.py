#!/usr/bin/env python3

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

import argparse
import os

import cv2
import numpy as np
import yaml
from tqdm import tqdm

import tensorflow as tf
from training.callbacks import ClassificationImages, SeekerImages
from training.dataset import Dataset
from training.flavour import get_flavour
from training.op import unmap_visual_mesh

if __name__ == "__main__":

    # Parse our command line arguments
    command = argparse.ArgumentParser(description="Utility for training a Visual Mesh network")
    command.add_argument("config", action="store", help="Path to the configuration file for training")

    args = command.parse_args()

    # Load our yaml file and convert it to an object
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for d in tf.data.TFRecordDataset(config["dataset"]["training"]["paths"]):
        first = tf.train.Example.FromString(d.numpy())
        print([k for k in first.features.feature])
        break

    datasets, loss, metrics, callbacks = get_flavour(config, "output")
    training_dataset, validation_dataset, _ = datasets

    scale = config["label"]["config"]["scale"]

    # colours = [c["colours"][0] for c in config["label"]["config"]["classes"]]
    for i, data in tqdm(enumerate(training_dataset)):

        # Work out the valid data ranges for each of the objects
        cs = [0] + np.cumsum(data["n"]).tolist()
        ranges = list(zip(cs, cs[1:]))

        nm = tf.concat(
            [
                unmap_visual_mesh(
                    data["V"][r[0] : r[1]],
                    height=data["Hoc"][i][0][2, 3],
                    model="XYGRID6",
                    geometry="SPHERE",
                    radius=0.0949996,
                )
                for i, r in enumerate(ranges)
            ],
            axis=0,
        )

        lens = {
            "projection": tf.reshape(data["lens"]["projection"], (-1,)),
            "focal_length": tf.reshape(data["lens"]["focal_length"], (-1,)),
            "centre": tf.reshape(data["lens"]["centre"], (-1, 2)),
            "k": tf.reshape(data["lens"]["k"], (-1, 2)),
            "fov": tf.reshape(data["lens"]["fov"], (-1,)),
        }

        for i, r in enumerate(ranges):
            img = callbacks[0].image(
                img=data["jpg"][i][0].numpy(),
                X=data["Y"][r[0] : r[1]],
                Hoc=data["Hoc"][i][0],
                lens={k: v[i] for k, v in lens.items()},
                nm=nm[r[0] : r[1]],
            )

            cv2.imwrite("out{}.jpg".format(i), img[1].numpy())

        exit(0)

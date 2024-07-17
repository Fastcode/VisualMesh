# Copyright (C) 2017-2024 Trent Houliston <trent@houliston.me>
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

import os

import numpy as np
import yaml

import tensorflow as tf

from .flavour import Dataset
from .layer.graph_convolution import GraphConvolution
from .model import VisualMeshModel


def unexport(config, output_path):

    with open(os.path.join(output_path, "model.yaml"), "r") as f:
        network = yaml.safe_load(f)

    # Get the training dataset so we know the output size
    training_dataset = Dataset(config, "training")

    # Get the dimensionality of the Y part of the dataset
    output_dims = training_dataset.element_spec["Y"].shape[-1]

    # Define the model
    model = VisualMeshModel(structure=config["network"]["structure"], output_dims=output_dims)

    # We have to run a predict step so that everything is loaded properly
    for v in training_dataset.take(1):
        model(v["X"], v["G"], training=False)

    stages = []
    for m in model.stages:
        op = model.ops[m]

        if type(op[0]) is GraphConvolution:
            op = op[0]
            stages.append(
                [
                    {
                        "weights": op.dense.weights[0],
                        "biases": op.dense.weights[1],
                        "activation": op.dense.activation.__name__,
                    }
                ]
            )
        elif type(op[0]) is tf.keras.layers.Dense:
            op = op[0]
            stages[-1].append(
                {
                    "weights": op.weights[0],
                    "biases": op.weights[1],
                    "activation": op.activation.__name__,
                }
            )
        else:
            print("Error: currently we can only import GraphConvolution and Dense layers")
            exit(1)

    for stage, conf in zip(stages, network["network"]):
        for w, c in zip(stage, conf):

            weight_shape = w["weights"].shape
            bias_shape = w["biases"].shape

            in_weights = tf.constant(c["weights"], w["weights"].dtype)
            in_biases = tf.constant(c["biases"], w["biases"].dtype)

            w["weights"].assign(in_weights[0 : weight_shape[0], 0 : weight_shape[1]])
            w["biases"].assign(in_biases[0 : bias_shape[0]])

    model.save_weights(os.path.join(output_path, "checkpoint"))

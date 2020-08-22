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

import os

import numpy as np
import tensorflow as tf
import yaml

from .dataset import keras_dataset
from .flavour import Dataset
from .layer.graph_convolution import GraphConvolution
from .model import VisualMeshModel


def export(config, output_path):

    # Get the training dataset so we know the output size
    training_dataset = (
        Dataset(config, "training")
        .map(keras_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    # Get the dimensionality of the Y part of the dataset
    output_dims = training_dataset.element_spec[1].shape[1]

    # Define the model
    model = VisualMeshModel(structure=config["network"]["structure"], output_dims=output_dims)

    # Find the latest checkpoint file and load it
    checkpoint_file = tf.train.latest_checkpoint(output_path)
    if checkpoint_file is not None:
        model.load_weights(checkpoint_file)
    else:
        raise RuntimeError("Could not find weights to load into the network")

    # We have to run a predict step so that everything is loaded properly
    model.predict(training_dataset.take(1))

    stages = []
    for m in model.stages:
        op = model.ops[m]

        if type(op[0]) is GraphConvolution:
            stages.append([])
        elif type(op[0]) is tf.keras.layers.Dense:
            op = op[0]
            stages[-1].append(
                {
                    "weights": op.weights[0].numpy().tolist(),
                    "biases": op.weights[1].numpy().tolist(),
                    "activation": op.activation.__name__,
                }
            )
        else:
            print("Error: currently we can only export GraphConvolution and Dense layers")
            exit(1)

    # While we have a 3 values on our input, all the c++ take 4 due to alignment issues
    # Therefore for that weights we need to increase it to 4
    first = tf.convert_to_tensor(stages[0][0]["weights"])
    first = tf.reshape(first, (-1, 3, first.shape[-1]))
    first = tf.pad(first, [[0, 0], [0, 1], [0, 0]])
    first = tf.reshape(first, (-1, first.shape[-1]))
    stages[0][0]["weights"] = first.numpy().tolist()

    with open(os.path.join(output_path, "model.yaml"), "w") as out:
        yaml.dump(stages, out)

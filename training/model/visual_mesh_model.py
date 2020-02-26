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

import tensorflow as tf

from training.layer import GraphConvolution


class VisualMeshModel(tf.keras.Model):
    def __init__(self, structure, n_classes, activation):
        super(VisualMeshModel, self).__init__()

        self.stages = []

        # Build our network structure
        for c in structure:

            # Graph convolution
            self.stages.append(GraphConvolution())

            # Dense internal layers
            for units in c:
                self.stages.append(
                    tf.keras.layers.Dense(
                        units=units,
                        activation=activation,
                        kernel_initializer="lecun_normal",
                        bias_initializer="lecun_normal",
                    )
                )

        # Final dense layer for the number of classes
        self.stages.append(
            tf.keras.layers.Dense(
                units=n_classes,
                activation=tf.nn.softmax,
                kernel_initializer="glorot_normal",
                bias_initializer="glorot_normal",
            )
        )

    def call(self, X, training=False):

        # Split out the graph and logits
        logits, G = X

        # Run through each of our layers in sequence
        for l in self.stages:
            if isinstance(l, GraphConvolution):
                logits = l(logits, G)
            elif isinstance(l, tf.keras.layers.Dense):
                logits = l(logits)

        # At the very end of the network, we remove the offscreen point
        logits = logits[:-1]

        return logits

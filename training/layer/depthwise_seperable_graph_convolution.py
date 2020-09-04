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


class Depthwise(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Depthwise, self).__init__()
        self.pointwise = tf.keras.layers.Dense(**kwargs)

    def build(self, input_shape):
        # Copy whatever we have on our pointwise kernel
        self.depthwise_weights = self.add_weight(
            "depthwise_kernel",
            input_shape[1:],
            dtype=self.dtype,
            initializer=self.pointwise.kernel_initializer,
            regularizer=self.pointwise.kernel_regularizer,
            constraint=self.pointwise.kernel_constraint,
        )

    def call(self, X):
        depthwise = tf.einsum("ijk,jk->ik", X, self.depthwise_weights)
        return self.pointwise(depthwise)


class DepthwiseSeparableGraphConvolution(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DepthwiseSeparableGraphConvolution, self).__init__()
        self.depthwise = Depthwise(**kwargs)

    def call(self, X, G):
        convolved = tf.reshape(tf.gather(X, G, name="NetworkGather"), shape=[-1, G.shape[-1], X.shape[-1]])
        return self.depthwise(convolved)

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

import math

import tensorflow as tf


def random_axis():
    # Get two random values between 0 and 1
    uv = tf.random.uniform([2], 0, 1)

    # Math to make a uniform distribution on the unit sphere
    # https://mathworld.wolfram.com/SpherePointPicking.html
    theta = 2 * math.pi * uv[0]
    phi = tf.math.acos(2.0 * uv[1] - 1.0)

    # Convert this into an axis
    return [tf.math.sin(theta) * tf.math.cos(phi), tf.math.sin(theta) * tf.math.sin(phi), tf.math.cos(theta)]


def random_rotation(mean, stddev):

    # Get a random axis to perform the rotation around
    ux, uy, uz = random_axis()

    # Get a random angle within our specifications
    angle = tf.random.truncated_normal((), mean, stddev)
    ca = tf.math.cos(angle)
    ia = 1.0 - ca
    sa = tf.math.sin(angle)

    # Axis angle to rotation matrix
    return tf.convert_to_tensor(
        [
            [ux * ux * ia + ca, uy * ux * ia - uz * sa, uz * ux * ia + uy * sa, 0.0],
            [ux * uy * ia + uz * sa, uy * uy * ia + ca, uz * uy * ia - ux * sa, 0.0],
            [ux * uz * ia - uy * sa, uy * uz * ia + ux * sa, uz * uz * ia + ca, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

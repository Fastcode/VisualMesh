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


def SeekerLoss():
    def seeker_loss(y_true, y_pred, sample_weight=None):

        # 0->0.5 normal loss, 0.5->0.75 signed/unsigned blend, 0.75+ clipped loss
        signed_end = 0.5
        unsigned_end = 0.75

        # Y should be between -1 and 1 but from the dataset it's the full value
        y_true = tf.clip_by_value(y_true, -1.0, 1.0)

        # Calculate the loss given signed and unsigned differences
        # As we get further from the origin we care less about the direction to near nearest object
        # and more about the fact that it is more distant than we can measure
        signed_loss = tf.math.squared_difference(y_true, y_pred)
        unsigned_loss = tf.math.squared_difference(tf.math.abs(y_true), tf.math.abs(y_pred))

        # We blend the signed and unsigned loss together in order to make sign less important with distance
        # Value will go from 0 when signed, to 1 when unsigned
        loss_factor = tf.clip_by_value(tf.divide(tf.math.abs(y_pred) - signed_end, unsigned_end - signed_end), 0.0, 1.0)
        near_loss = tf.multiply(signed_loss, 1.0 - loss_factor) + tf.multiply(unsigned_loss, loss_factor)

        # Once we exceed the unsigned region we don't care how far we are from the object we just care that we are "far"
        # So for loss we will try to push to the edge of the unsigned region and if we are greater, loss is 0
        far_loss = tf.where(tf.math.abs(y_pred) > unsigned_end, tf.zeros_like(y_pred), unsigned_loss)

        # We apply different losses depending on the label of both x and y
        # If one of our losses is far but not both, we ignore the non far loss
        # | X | Y | XL | YL |
        # |---|---|----|----|
        # | N | N | NL | NL |
        # | N | F | IL | FL |
        # | F | N | FL | IL |
        # | F | F | FL | FL |

        # Select the correct loss function depending on location
        near_mean = tf.reduce_mean(near_loss, axis=-1)
        far_mean = tf.reduce_mean(far_loss, axis=-1)
        near = tf.math.abs(y_true) <= unsigned_end
        loss = tf.where(
            tf.reduce_any(near, axis=-1),
            tf.where(tf.reduce_all(near, axis=-1), near_mean, tf.where(near[:, 0], far_loss[:, 1], far_loss[:, 0])),
            far_mean,
        )

        return loss

    return seeker_loss

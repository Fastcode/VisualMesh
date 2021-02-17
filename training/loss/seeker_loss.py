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

        # Y true contains a list of all the possible answers, we are ranked on how close we are to the closest one
        # This code works out the closest point to our prediction for each case so we can use that for training
        y_true = tf.gather_nd(
            y_true,
            tf.stack(
                [
                    tf.range(tf.shape(y_true)[0], dtype=tf.int32),
                    tf.math.argmin(
                        tf.reduce_sum(tf.math.squared_difference(tf.expand_dims(y_pred, axis=-2), y_true), axis=-1),
                        axis=-1,
                        output_type=tf.int32,
                    ),
                ],
                axis=1,
            ),
        )

        # 0->0.5 normal loss, 0.5->0.75 signed/unsigned blend, 0.75+ clipped loss
        signed_end = 0.5
        unsigned_end = 0.75

        # Y should be between -1 and 1 but from the dataset it's the full scale value
        y_true = tf.clip_by_value(y_true, -1.0, 1.0)

        # Calculate the loss for when the point is not far away
        # from 0.0->signed_end we use the squared error so we get an accurate prediction of the result
        # from signed_end->unsigned_end we use squared error of the absolute values so magnitude is more important than sign
        loss_factor = tf.clip_by_value(tf.divide(tf.math.abs(y_true) - signed_end, unsigned_end - signed_end), 0.0, 1.0)
        near_loss = tf.reduce_mean(
            tf.add(
                tf.multiply(tf.math.squared_difference(y_true, y_pred), 1.0 - loss_factor),
                tf.multiply(tf.math.squared_difference(tf.math.abs(y_true), tf.math.abs(y_pred)), loss_factor),
            ),
            axis=-1,
        )

        # For far loss, if either value is greater than unsigned_end
        # then we check that one of the predictions is greater than unsigned end
        far_loss = tf.math.squared_difference(
            tf.reduce_max(tf.abs(y_true), axis=-1), tf.reduce_max(tf.abs(y_pred), axis=-1)
        )

        # For our far loss, if we predict over unsigned_end then set the loss for this to 0 (correctly predicted)
        far_loss = tf.where(tf.reduce_max(tf.abs(y_pred), axis=-1) < unsigned_end, far_loss, tf.zeros_like(far_loss))

        # Based on our categories create each of the three loss zones
        loss = tf.reduce_mean(
            tf.where(
                tf.reduce_all(tf.abs(y_true) < unsigned_end, axis=-1), tf.reduce_mean(near_loss, axis=-1), far_loss
            )
        )

        return loss

    return seeker_loss

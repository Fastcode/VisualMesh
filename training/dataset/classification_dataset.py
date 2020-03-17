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
from .visual_mesh_dataset import VisualMeshDataset


class ClassificationDataset(VisualMeshDataset):
    def __init__(self, classes, **kwargs):
        super(ClassificationDataset, self).__init__(features={"mask": tf.io.FixedLenFeature([], tf.string)}, **kwargs)
        self.classes = classes

    def get_labels(self, args):

        # Use the nearest neighbour pixel to get the classification from the mask
        Y = tf.gather_nd(tf.image.decode_image(args["mask"], channels=4), tf.cast(tf.round(args["px"]), tf.int32))

        # Expand the classes from colours into individual columns
        W = tf.image.convert_image_dtype(Y[:, 3], tf.float32)  # Alpha channel
        cs = []
        for c in self.classes:
            cs.append(
                tf.where(
                    tf.logical_and(
                        tf.reduce_any(
                            tf.stack(
                                [tf.reduce_all(input_tensor=tf.equal(Y[:, :3], [v]), axis=-1) for v in c["colours"]],
                                axis=-1,
                            ),
                            axis=-1,
                        ),
                        tf.greater(W, 0.0),
                    ),
                    1.0,
                    0.0,
                )
            )
        Y = tf.stack(cs, axis=-1)

        return Y, W

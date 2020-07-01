# Copyright (C) 2017-2020 Alex Biddulph <Alexander.Biddulph@uon.edu.au>
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

from training.op import link_nearest


class Stereoscopic:
    def __init__(self, projection, **config):
        self.geometry = projection["config"]["geometry"]["shape"]
        self.radius = projection["config"]["geometry"]["radius"]
        self.intersections = projection["config"]["geometry"]["intersections"]

    def prefixes(self):
        return ("left/", "right/")

    def merge(self, views):
        # Take the left/ and right/ prefix and perform the merging operation to convert them into a single output

        # {
        #     "X": X,        (n, 3)          pixel data
        #     "Y": Y,        (n, classes)    Class IDs
        #     "G": G,        (n, graph size) Graph connections (indexes into X)
        #     "n": n,        (scalar)        Number of graph nodes
        #     "C": C,        (n, 2)          Pixel coordinates
        #     "V": V,        (n, 3)          Unit vectors
        #     "Hoc": Hoc,    (4, 4)          Camera to observation plane homogeneous transform
        #     "jpg": jpg,                    Raw compressed image bytes
        #     "image":                       Decoded image
        #     "lens/projection":             Lens projection
        #     "lens/focal_length":           Lens focal length
        #     "lens/centre":                 Lens centre offset
        #     "lens/k":                      Lens distortion parameters
        #     "lens/fov":                    Lens fov
        # }

        # Find the nearest links in both graphs
        # For each vector in v_a find the vector in v_b that is radially closest to it.
        # Returned tensor has
        #   - same shape as v_a
        #   - indexes into v_b
        #   - should be concatenated on g_a
        left_matches = link_nearest(
            v_a=views["left/"]["V"],
            v_b=views["right/"]["V"],
            g_b=views["right/"]["G"],
            hoc_a=views["left/"]["Hoc"],
            hoc_b=views["right/"]["Hoc"],
            geometry=self.geometry,
            radius=self.radius,
            distance_threshold=1.0 / (1.5 * self.intersections),
        )
        right_matches = link_nearest(
            v_a=views["right/"]["V"],
            v_b=views["left/"]["V"],
            g_b=views["left/"]["G"],
            hoc_a=views["right/"]["Hoc"],
            hoc_b=views["left/"]["Hoc"],
            geometry=self.geometry,
            radius=self.radius,
            distance_threshold=1.0 / (1.5 * self.intersections),
        )

        # Cocatenate left_matches on to left_graph
        # Add a offset to the indices as the right pixel data are appended to the end of the list
        G_left = tf.concat([views["left/"]["G"], tf.shape(views["left/"]["G"])[0] + left_matches], axis=1)

        # Cocatenate right_matches on to right_graph
        # Offset needed on the right graph this time
        G_right = tf.concat([tf.shape(views["left/"]["G"])[0] + views["right/"]["G"], right_matches], axis=1)

        # Concatenate data
        G = tf.concat([G_left, G_right], axis=0, name="stereoscopic/merge/concat/G")
        return {
            **{
                k: tf.stack(
                    [views["left/"][k], views["right/"][k]], axis=0, name="stereoscopic/merge/stack/{}".format(k)
                )
                for k in views["left/"]
            },
            "X": tf.concat([views["left/"]["X"], views["right/"]["X"]], axis=0, name="stereoscopic/merge/concat/X"),
            "Y": tf.concat([views["left/"]["Y"], views["right/"]["Y"]], axis=0, name="stereoscopic/merge/concat/Y"),
            "G": G,
            "C": tf.concat([views["left/"]["C"], views["right/"]["C"]], axis=0, name="stereoscopic/merge/concat/C"),
            "V": tf.concat([views["left/"]["V"], views["right/"]["V"]], axis=0, name="stereoscopic/merge/concat/V"),
            "n": tf.stack([tf.shape(G_left)[0], tf.shape(G_right)[0]], axis=0, name="stereoscopic/merge/stack/n"),
        }

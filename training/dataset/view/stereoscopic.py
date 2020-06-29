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

from training.op import align_graphs


class Stereoscopic:
    def __init__(self, **config):
        self.geometry = config["projection"]["config"]["geometry"]["shape"]
        self.radius = config["projection"]["config"]["geometry"]["radius"]
        self.intersections = config["projection"]["config"]["geometry"]["intersections"]

    def prefixes(self):
        return ("left/", "right/")

    def merge(self, views):
        # Take the left/ and right/ prefix and perform the merging operation to convert them into a single output

        # return {
        #     "X": X,        (n + 1, 3)          pixel data
        #     "Y": Y,        (n, classes)        Class IDs
        #     "G": G,        (n + 1, graph size) Graph connections (indexes into X)
        #     "n": n,        (scalar)            Number of graph nodes
        #     "C": C,        (n, 2)              Pixel coordinates
        #     "V": V,        (n, 3)              Unit vectors
        #     "Hoc": Hoc,
        #     "jpg": jpg,
        #     "lens": lens,
        # }

        # Align the graphs
        matched_left = align_graphs(
            v_a=views["left/"]["V"],
            v_b=views["right/"]["V"],
            g_b=views["right/"]["G"],
            Hoc_a=views["left/"]["Hoc"],
            Hoc_b=views["right/"]["Hoc"],
            geometry=self.geometry,
            radius=self.radius,
            distance_threshold=1.0 / (1.5 * self.intersections),
        )
        matched_right = align_graphs(
            v_a=views["right/"]["V"],
            v_b=views["left/"]["V"],
            g_b=views["left/"]["G"],
            Hoc_a=views["right/"]["Hoc"],
            Hoc_b=views["left/"]["Hoc"],
            geometry=self.geometry,
            radius=self.radius,
            distance_threshold=1.0 / (1.5 * self.intersections),
        )

        raise RuntimeError("Stereoscopic is not yet implemented")

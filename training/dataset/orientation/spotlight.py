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


class Spotlight:
    def __init__(self, **config):
        pass

    def features(self):
        return {
            "Hoc": tf.io.FixedLenFeature([4, 4], tf.float32),
            "target": tf.io.FixedLenSequenceFeature([3], tf.float32),
        }

    def __call__(self, Hoc, target, **features):
        # Create a new Hcw via target and the original Hcw

        # Apply variations to the position of the target

        # World up is still world up so therefore world z = our z
        # z = normalise(target vector)
        # x = normalise(world z cross target)
        # y = normalise(x cross z)
        # Height = norm target vector
        raise RuntimeError("Spotlight is not yet implemented")

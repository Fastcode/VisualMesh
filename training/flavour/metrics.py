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

from ..metrics import *


def Metrics(config):

    if config["label"]["type"] == "Classification":

        classes = config["label"]["config"]["classes"]
        metrics = [
            AveragePrecision("metrics/average_precision", len(classes)),
            AverageRecall("metrics/average_recall", len(classes)),
        ]
        for i, c in enumerate(classes):
            metrics.append(ClassPrecision("metrics/{}_precision".format(c["name"]), i, len(classes)))
            metrics.append(ClassRecall("metrics/{}_recall".format(c["name"]), i, len(classes)))

        return metrics

    elif config["label"]["type"] == "Seeker":
        return [
            SeekerPrecision("metrics/precision75", 0.75),
            SeekerRecall("metrics/recall75", 0.75),
            SeekerStdDev("metrics/stddev75", 0.75),
            SeekerPrecision("metrics/precision50", 0.5),
            SeekerRecall("metrics/recall50", 0.5),
            SeekerStdDev("metrics/stddev50", 0.5),
        ]

    else:
        raise RuntimeError("Cannot create metrics, {} is not a supported  type".format(config["label"]["type"]))

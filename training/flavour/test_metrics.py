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

import training.metrics.test.confusion_curve as confusion
from training.metrics.test import Confusion, ConfusionCurve
from training.metrics.test.seeker_hourglass import SeekerHourglass


def TestMetrics(config):

    if config["label"]["type"] == "Classification":
        classes = config["label"]["config"]["classes"]
        curves = [Confusion(name="metrics/confusion", classes=classes)]

        for i, c in enumerate(classes):
            curves.extend(
                [
                    ConfusionCurve(
                        name="metrics/curves/{}_pr".format(c["name"]),
                        class_index=i,
                        x_axis=confusion._tpr,
                        y_axis=confusion._ppv,
                        sort_axis=confusion._threshold,
                        chart={
                            "title": "{} Precision/Recall".format(c["name"].title()),
                            "x_label": "Recall",
                            "y_label": "Precision",
                            "sort_label": "Threshold",
                        },
                    ),
                    ConfusionCurve(
                        name="metrics/curves/{}_roc".format(c["name"]),
                        class_index=i,
                        x_axis=confusion._fpr,
                        y_axis=confusion._tpr,
                        sort_axis=confusion._threshold,
                        chart={
                            "title": "{} Receiver Operating Characteristics".format(c["name"].title()),
                            "x_label": "False Positive Rate",
                            "y_label": "True Positive Rate",
                            "sort_label": "Threshold",
                        },
                    ),
                    ConfusionCurve(
                        name="metrics/curves/{}_mi".format(c["name"]),
                        class_index=i,
                        x_axis=confusion._markedness,
                        y_axis=confusion._informedness,
                        sort_axis=confusion._threshold,
                        chart={
                            "title": "{} Informedness/Markedness".format(c["name"]),
                            "x_label": "Markedness",
                            "y_label": "Informedness",
                            "sort_label": "Threshold",
                        },
                    ),
                    ConfusionCurve(
                        name="metrics/curves/{}_precision".format(c["name"]),
                        class_index=i,
                        x_axis=confusion._threshold,
                        y_axis=confusion._ppv,
                        sort_axis=confusion._threshold,
                        chart={
                            "title": "{} Precision".format(c["name"]),
                            "x_label": "Threshold",
                            "y_label": "Precision",
                            "sort_label": "Threshold",
                        },
                    ),
                    ConfusionCurve(
                        name="metrics/curves/{}_recall".format(c["name"]),
                        class_index=i,
                        x_axis=confusion._threshold,
                        y_axis=confusion._tpr,
                        sort_axis=confusion._threshold,
                        chart={
                            "title": "{} Recall".format(c["name"]),
                            "x_label": "Threshold",
                            "y_label": "Recall",
                            "sort_label": "Threshold",
                        },
                    ),
                    ConfusionCurve(
                        name="metrics/curves/{}_f1".format(c["name"]),
                        class_index=i,
                        x_axis=confusion._threshold,
                        y_axis=confusion._f1,
                        sort_axis=confusion._threshold,
                        chart={
                            "title": "{} F1 Score".format(c["name"]),
                            "x_label": "Threshold",
                            "y_label": "F1 Score",
                            "sort_label": "Threshold",
                        },
                    ),
                    ConfusionCurve(
                        name="metrics/curves/{}_informedness".format(c["name"]),
                        class_index=i,
                        x_axis=confusion._threshold,
                        y_axis=confusion._informedness,
                        sort_axis=confusion._threshold,
                        chart={
                            "title": "{} Informedness".format(c["name"]),
                            "x_label": "Threshold",
                            "y_label": "Informedness",
                            "sort_label": "Threshold",
                        },
                    ),
                    ConfusionCurve(
                        name="metrics/curves/{}_markedness".format(c["name"]),
                        class_index=i,
                        x_axis=confusion._threshold,
                        y_axis=confusion._markedness,
                        sort_axis=confusion._threshold,
                        chart={
                            "title": "{} Markedness".format(c["name"]),
                            "x_label": "Threshold",
                            "y_label": "Markedness",
                            "sort_label": "Threshold",
                        },
                    ),
                    ConfusionCurve(
                        name="metrics/curves/{}_mcc".format(c["name"]),
                        class_index=i,
                        x_axis=confusion._threshold,
                        y_axis=confusion._mcc,
                        sort_axis=confusion._threshold,
                        chart={
                            "title": "{} $\phi$ Coefficent".format(c["name"]),
                            "x_label": "Threshold",
                            "y_label": "$\phi$ Coefficent",
                            "sort_label": "Threshold",
                        },
                    ),
                ]
            )

        return curves

    elif config["label"]["type"] == "Seeker":
        return [
            SeekerHourglass(name="metrics/curves/hourglass"),
        ]

    else:
        raise RuntimeError("Cannot create loss function, {} is not a supported type".format(config["label"]["type"]))

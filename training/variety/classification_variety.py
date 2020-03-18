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

from ..dataset import ClassificationDataset, keras_dataset
from ..loss import FocalLoss
from ..callbacks import ClassificationImages
from ..metrics import AverageRecall, AveragePrecision, ClassPrecision, ClassRecall


def classification_variety(config, output_path):

    # Get some arguments that will always be added to datasets
    dataset_args = {
        "mesh": config["model"]["mesh"],
        "geometry": config["model"]["geometry"],
        "prefetch": tf.data.experimental.AUTOTUNE,
    }
    training_dataset_args = {
        **dataset_args,
        **config["dataset"]["training"],
        "batch_size": config["training"]["batch_size"],
    }
    validation_dataset_args = {
        **dataset_args,
        **config["dataset"]["validation"],
        "batch_size": config["training"]["validation"]["batch_size"],
    }
    testing_dataset_args = {
        **dataset_args,
        **config["dataset"]["validation"],
        "batch_size": config["testing"]["batch_size"],
    }

    # Load classification datasets
    training_dataset = (
        ClassificationDataset(**training_dataset_args, classes=config["dataset"]["output"]["classes"])
        .build()
        .map(keras_dataset)
    )
    validation_dataset = (
        ClassificationDataset(**validation_dataset_args, classes=config["dataset"]["output"]["classes"])
        .build()
        .map(keras_dataset)
    )
    testing_dataset = (
        ClassificationDataset(**testing_dataset_args, classes=config["dataset"]["output"]["classes"])
        .build()
        .map(keras_dataset)
    )

    # Use focal loss for classification tasks
    loss = FocalLoss()

    # Metrics that we want to track
    output_dims = training_dataset.element_spec[1].shape[1]
    metrics = [
        AveragePrecision("metrics/average_precision", output_dims),
        AverageRecall("metrics/average_recall", output_dims),
    ]
    for i, k in enumerate(config["dataset"]["output"]["classes"]):
        metrics.append(ClassPrecision("metrics/{}_precision".format(k["name"]), i, output_dims))
        metrics.append(ClassRecall("metrics/{}_recall".format(k["name"]), i, output_dims))

    # Classification image callback
    callbacks = [
        ClassificationImages(
            output_path,
            config["dataset"]["validation"]["paths"],
            config["dataset"]["output"]["classes"],
            config["model"]["mesh"],
            config["model"]["geometry"],
            config["training"]["validation"]["progress_images"],
            # Draw using the first colour in the list
            [c["colours"][0] for c in config["dataset"]["output"]["classes"]],
        )
    ]

    return (training_dataset, validation_dataset, testing_dataset), loss, metrics, callbacks

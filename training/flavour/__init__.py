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

from training.callbacks import *
from training.dataset import Dataset
from training.loss import *
from training.metrics import *


def _merge_flavour_config(base, detail):
    def _merge(a, b):

        # If they're both dictionaries we start with a and then overwrite with b
        if type(a) == dict and type(b) == dict:
            v = {**a}
            for k in b:
                if k not in v:
                    v[k] = b[k]
                else:
                    v[k] = _merge(a[k], b[k])
            return v

        # Otherwise b always wins
        return b

    config = {}
    for k in ["view", "example", "orientation", "label", "projection"]:
        if k not in detail:
            config[k] = base[k]
        else:
            config[k] = _merge(base[k], detail[k])

    return config


def get_flavour(config, output_path):

    training = config["dataset"]["training"]
    validation = config["dataset"]["validation"]
    testing = config["dataset"]["testing"]

    training_config = _merge_flavour_config(config, training.get("config", {}))
    validation_config = _merge_flavour_config(config, validation.get("config", {}))
    testing_config = _merge_flavour_config(config, testing.get("config", {}))

    # Datasets work out the flavour themselves
    datasets = (
        Dataset(
            paths=training["paths"],
            batch_size=training["batch_size"],
            keys=training.get("keys", {}),
            **training_config,
        ),
        Dataset(
            paths=validation["paths"],
            batch_size=validation["batch_size"],
            keys=validation.get("keys", {}),
            **validation_config,
        ).repeat(),
        Dataset(
            paths=testing["paths"], batch_size=testing["batch_size"], keys=testing.get("keys", {}), **testing_config,
        ),
    )

    # Flavour
    if config["label"]["type"] == "Classification":
        classes = validation_config["label"]["config"]["classes"]

        # Classification uses focal loss
        loss = FocalLoss()

        # Class metrics
        output_dims = datasets[0].element_spec["Y"].shape[1]
        metrics = [
            AveragePrecision("metrics/average_precision", output_dims),
            AverageRecall("metrics/average_recall", output_dims),
        ]
        for i, c in enumerate(classes):
            metrics.append(ClassPrecision("metrics/{}_precision".format(c["name"]), i, output_dims))
            metrics.append(ClassRecall("metrics/{}_recall".format(c["name"]), i, output_dims))

        # Callbacks
        callbacks = [
            ClassificationImages(
                output_path=output_path,
                dataset=Dataset(
                    paths=validation["paths"],
                    batch_size=config["training"]["validation"]["progress_images"],
                    keys=validation.get("keys", {}),
                    **validation_config,
                ).take(1),
                # Draw using the first colour in the list for each class
                colours=[c["colours"][0] for c in classes],
            )
        ]

    if config["label"]["type"] == "Seeker":

        # We use seeker loss which is based around a tanh style pointer
        loss = SeekerLoss()

        # TODO metric that is the mean squared error
        # TODO metric that describes our "precision" and "recall" (that being that when we are close we get a non 1.0 value and when we are far we get a 1.0 value)
        metrics = []

        callbacks = [
            SeekerImages(
                output_path=output_path,
                dataset=Dataset(
                    paths=validation["paths"],
                    batch_size=config["training"]["validation"]["progress_images"],
                    keys=validation.get("keys", {}),
                    **validation_config,
                ).take(1),
                model=validation_config["projection"]["config"]["mesh"]["model"],
                geometry=validation_config["projection"]["config"]["geometry"]["shape"],
                radius=validation_config["projection"]["config"]["geometry"]["radius"],
                scale=validation_config["label"]["config"]["scale"],
            )
        ]

    # Callbacks
    return datasets, loss, metrics, callbacks

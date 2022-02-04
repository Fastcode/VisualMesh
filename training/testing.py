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

from tqdm import tqdm

import tensorflow as tf

from .flavour import Dataset, Loss, TestMetrics
from .model import VisualMeshModel


def test(config, output_path):

    # Get the testing dataset
    testing_dataset = Dataset(config, "testing")
    # Get the dimensionality of the Y part of the dataset
    output_dims = testing_dataset.element_spec["Y"].shape[-1]

    # Define the model
    model = VisualMeshModel(structure=config["network"]["structure"], output_dims=output_dims)

    # Find the latest checkpoint file and load it
    checkpoint_file = tf.train.latest_checkpoint(output_path)
    if checkpoint_file is not None:
        model.load_weights(checkpoint_file)
    else:
        raise RuntimeError("Could not find weights to load into the network")

    # Get the metrics and curves we will be building
    metrics = TestMetrics(config)
    loss_fn = Loss(config)

    loss_sum = 0
    loss_count = 0
    for data in tqdm(testing_dataset, desc="Testing", unit="batch", dynamic_ncols=True, leave=False):
        logits = model(data["X"], data["G"], training=False)
        loss_sum += loss_fn(data["Y"], logits)
        loss_count += 1

        for m in metrics:
            m.update_state(data["Y"], logits)

    # Save all the metric data
    for m in metrics:
        m.save(output_path)

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

import os

from tqdm import tqdm, trange

import tensorflow as tf
from training import dataset

from .flavour import Dataset, LearningRate, Loss, Metrics, Optimiser, ProgressImages
from .model import VisualMeshModel


# Writes out the metrics, handling the different kinds of metrics that we could have
def _write_metrics(writer, step, loss, metrics, prefix="", images=True, reset=False):
    m_status = [f"Loss: {loss:.02g}"]

    with writer.as_default():
        tf.summary.scalar(f"{prefix}loss", loss, step)
        for m in metrics:
            if hasattr(m, "images"):
                if images:
                    tf.summary.image(f"{prefix}{m.name}", m.images(m.result()), step)
            else:
                r = m.result()
                m_status.append(f"{m.name}: {r:.02g}")
                tf.summary.scalar(f"{prefix}{m.name}", r, step)

    return m_status


class DatasetGrouper:
    def __init__(self, dataset, group_size):
        self.dataset = dataset
        self.it = iter(dataset)
        self.group_size = group_size

    def reset(self):
        self.it = iter(dataset)

    def __len__(self):
        if self.group_size is None:
            return len(self.dataset)
        else:
            return self.group_size

    def __iter__(self):
        # Entire dataset as group
        if self.group_size is None:
            for v in self.dataset:
                yield v
        else:
            for i in range(self.group_size):
                yield next(self.it)


# Train the network
def train(config, output_path):

    # Get the flavour of network that we are training and prepare them
    training_dataset = Dataset(config, "training")
    validation_dataset = Dataset(config, "validation")
    loss_fn = Loss(config)
    metrics = Metrics(config)
    learning_rate = LearningRate(config)
    optimiser = Optimiser(config, learning_rate(0))
    progress_images = ProgressImages(config, output_path)

    # Get the dimensionality of the Y part of the dataset so we know the size of the last layer of the network
    output_dims = training_dataset.element_spec["Y"].shape[-1]

    # Define the model
    model = VisualMeshModel(structure=config["network"]["structure"], output_dims=output_dims)

    # Create the tensorboard writers
    train_writer = tf.summary.create_file_writer(os.path.join(output_path, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(output_path, "validation"))

    # Find the latest checkpoint file and load it so we can resume training
    checkpoint_file = tf.train.latest_checkpoint(output_path)
    if checkpoint_file is not None:
        model.load_weights(checkpoint_file)

    # Create the iterator objects to loop through the dataset
    if config["training"]["batches"] is not None:
        training_dataset = training_dataset.repeat()
    validation_dataset = validation_dataset.repeat()
    training = DatasetGrouper(training_dataset, config["training"]["batches_per_epoch"])
    validation = DatasetGrouper(validation_dataset, config["training"]["validation"]["samples"])

    batch_no = 0
    best_loss = None
    for epoch in trange(config["training"]["epochs"], desc="Epoch", unit="epoch", dynamic_ncols=True):

        # Update the learning rate
        optimiser.lr = learning_rate(epoch)

        # Perform the training loop
        loss_sum = 0
        loss_count = 0
        for data in tqdm(training, desc="Train Batch", unit="batch", dynamic_ncols=True, leave=False):

            # Run the network
            with tf.GradientTape() as tape:
                logits = model(data["X"], data["G"], training=True)
                loss = loss_fn(data["Y"], logits)

            # If our loss ever becomes non finite end training
            if not tf.math.is_finite(loss):
                tqdm.write("Loss is not finite, terminating training")
                exit(1)

            # Apply the gradient update
            grads = tape.gradient(loss, model.trainable_weights)
            optimiser.apply_gradients(zip(grads, model.trainable_weights))

            # Update all the metric states
            for m in metrics:
                m.update_state(data["Y"], logits)
            loss_sum += loss
            loss_count += 1

            # If we are doing batch level logs
            if config["training"]["validation"]["log_frequency"] == "batch":
                _write_metrics(
                    writer=train_writer,
                    loss=loss,
                    step=batch_no,
                    prefix="batch/",
                    metrics=metrics,
                    images=False,
                    reset=False,
                )

            batch_no += 1

        # Output progress images
        progress_images(model, epoch)

        # Log epoch level stats
        train_metric = _write_metrics(
            writer=train_writer,
            loss=(loss_sum / loss_count),
            step=epoch,
            prefix="epoch/",
            metrics=metrics,
            images=True,
            reset=True,
        )

        tqdm.write("Training Epoch {}\n{}".format(epoch, ", ".join(train_metric)))

        # Run validation at the end of each step
        loss_sum = 0
        loss_count = 0
        for data in tqdm(validation, desc="Validation Batch", unit="batch", dynamic_ncols=True, leave=False):
            logits = model(data["X"], data["G"], training=False)
            loss_sum += loss_fn(data["Y"], logits)
            loss_count += 1

            for m in metrics:
                m.update_state(data["Y"], logits)

        # Validation stats
        validation_metric = _write_metrics(
            writer=val_writer,
            loss=loss_sum / loss_count,
            step=epoch,
            prefix="epoch/",
            metrics=metrics,
            images=True,
            reset=True,
        )

        tqdm.write("Validation Epoch {}\n{}".format(epoch, ", ".join(validation_metric)))

        # If this is the best model we have seen, save it
        if best_loss is None or best_loss > loss_sum / loss_count:
            model.save_weights(os.path.join(output_path, "checkpoint"))

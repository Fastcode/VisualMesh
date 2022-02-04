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

import math
import os
import pickle

import matplotlib as mpl
import numpy as np
from tqdm import tqdm

import tensorflow as tf

from .flavour import Dataset, Loss, Optimiser
from .model import VisualMeshModel

mpl.use("Agg")
import matplotlib.pyplot as plt  # isort:skip


# Find valid learning rates for
def find_lr(config, output_path, min_lr, max_lr, n_steps, window_size):

    # Open the training dataset and put it on repeat
    training_dataset = Dataset(config, "training").repeat()

    # Get the dimensionality of the Y part of the dataset
    output_dims = training_dataset.element_spec["Y"].shape[-1]

    # Define the model
    model = VisualMeshModel(structure=config["network"]["structure"], output_dims=output_dims)

    def lr_schedule(epoch):
        return min_lr * (max_lr / min_lr) ** (epoch / n_steps)

    optimiser = Optimiser(config, lr_schedule(0))
    loss_fn = Loss(config)

    losses = []
    lrs = []
    with tqdm(desc=f"LR: {optimiser.lr.numpy():.3e} Loss: ---", total=n_steps) as progress:
        for step, data in enumerate(training_dataset):
            if step > n_steps:
                break

            optimiser.lr = lr_schedule(step)

            with tf.GradientTape() as tape:
                logits = model(data["X"], data["G"], training=True)
                loss = loss_fn(data["Y"], logits)

            if not tf.math.is_finite(loss):
                break

            grads = tape.gradient(loss, model.trainable_weights)
            optimiser.apply_gradients(zip(grads, model.trainable_weights))

            losses.append(loss.numpy())
            lrs.append(optimiser.lr.numpy())

            progress.set_description(f"LR: {optimiser.lr.numpy():.3e} Loss: {loss.numpy():.3e}")
            progress.update()

    # Extract the loss and LR from before it became nan
    loss = np.array(losses[:-1])
    lr = np.array(lrs[:-1])

    # Make a smoothed version of the loss
    smooth_loss = np.convolve(loss, np.ones(window_size), "same") / np.convolve(
        np.ones_like(loss), np.ones(window_size), "same"
    )

    # Find the point where the loss goes statistically off for the first time
    # We then add half our average window since that is where the spike actually starts
    delta = np.log10(smooth_loss)[1:] - np.log10(smooth_loss)[:-1]
    exploding = (window_size // 2) + np.argmax(delta > 3.0 * np.std(delta))

    # Work out the suggested maximum learning rate
    suggested_max = lr[exploding] / 10
    print("Suggested maximum learning rate: {:.0e}".format(suggested_max))

    # Output the details to file
    os.makedirs(os.path.join(output_path, "lr_finder"), exist_ok=True)
    with open(os.path.join(output_path, "lr_finder", "max_lr.csv"), "w") as f:
        f.write("{:.0e}".format(suggested_max))

    np.savetxt(os.path.join(output_path, "lr_finder", "lr_loss.csv"), np.stack([lr, loss], axis=-1), delimiter=",")

    # Draw the plots of raw and smoothed data
    fig, ax = plt.subplots()
    ax.set_title("Learning Rate Finder")
    ax.set_xlabel("log(Learning Rate)")
    ax.set_ylabel("log(Loss)")
    ax.plot(np.log10(lr), np.log10(loss))
    ax.plot(np.log10(lr), np.log10(smooth_loss))
    fig.savefig(os.path.join(output_path, "lr_finder", "lr_loss.png"))
    plt.close(fig)

    # Draw the plots of the delta in the smoothed loss
    fig, ax = plt.subplots()
    ax.set_title("Learning Rate Finder")
    ax.set_xlabel("log(Learning Rate)")
    ax.set_ylabel("Δlog(Loss)")
    ax.plot(np.log10(lr)[:-1], delta)
    fig.savefig(os.path.join(output_path, "lr_finder", "delta_lr_loss.png"))
    plt.close(fig)

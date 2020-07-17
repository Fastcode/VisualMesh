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

import os
import numpy as np
import tensorflow as tf


# Based on the implementation at
# https://medium.com/dsnet/the-1-cycle-policy-an-experiment-that-vanished-the-struggle-in-training-neural-nets-184417de23b9
class FindLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, config, output_path, **kwargs):
        super(FindLearningRate, self).__init__()

        # The lower boundary for learning rate (initial lr)
        self.min_lr = float(config["min_lr"])
        # The upper boundary for learning rate
        self.max_lr = float(config["max_lr"])
        # Total number of iterations used for this test run (lr is calculated based on this)
        self.bn = int(config["num_iterations"]) - 1

        ratio = self.max_lr / self.min_lr  # n
        self.mult = ratio ** (1 / self.bn)  # q = (max_lr/init_lr)^(1/n)
        self.best_loss = 1e9  # our assumed best loss
        self.iteration = 0  # current iteration, initialized to 0
        self.lrs = []
        self.losses = []

        # Smoothed loss
        self.use_smoothed_loss = config["use_smoothed_loss"]
        self.running_loss = 0.0
        self.smoothed_loss = 0.0
        self.beta = config["exp_beta"]

        # Location to store CSV file
        self.output_file_path = os.path.join(output_path, "find_lr.csv")

    def calc_lr(self, loss):
        self.iteration += 1
        if np.isnan(loss) or loss > 4.0 * self.best_loss:  # stopping criteria (if current loss > 4*best loss)
            return -1

        # if current_loss < best_loss, replace best_loss with current_loss
        if loss < self.best_loss and self.iteration > 1:
            self.best_loss = loss

        mult = self.mult ** self.iteration  # q = q^i
        lr = self.min_lr * mult  # lr_i = init_lr * q
        self.lrs.append(lr)  # append the learing rate to lrs
        self.losses.append(loss)  # append the loss to losses

        return lr

    def dump_csv(self, output_file):
        with open(output_file, "w") as f:
            f.write("lr,log10 lr,loss\n")
            for lr, loss in zip(self.lrs, self.losses):
                f.write("{},{},{}\n".format(lr, np.log10(lr), loss))

    def on_train_begin(self, logs):
        # Make sure we are using the correct learning rate from the very beginning
        tf.keras.backend.set_value(self.model.optimizer.lr, self.min_lr)

        # Print hte CSV header to the output file
        with open(self.output_file_path, "w") as f:
            f.write("lr,log10 lr,loss\n")

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")

        if loss is not None and tf.math.is_finite(loss):
            if self.use_smoothed_loss:
                self.running_loss = self.beta * self.running_loss + (1 - self.beta) * loss
                self.smoothed_loss = self.running_loss / (1 - self.beta ** (self.iteration + 1))
                lr = self.calc_lr(self.smoothed_loss)
            else:
                lr = self.calc_lr(loss)

            # Time to stop
            if lr == -1 or lr > self.max_lr:
                self.model.stop_training = True
                return

            # Update learning rate
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)

            # Update logs
            logs["lr"] = lr

            with open(self.output_file_path, "a") as f:
                f.write("{},{},{}\n".format(self.lrs[-1], np.log10(self.lrs[-1]), self.losses[-1]))

        elif not tf.math.is_finite(loss):
            self.model.stop_training = True
            return

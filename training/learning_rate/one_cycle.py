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


class OneCycle:
    def __init__(self, config):

        # lr ranges
        self.min_lr = float(config["training"]["learning_rate"]["min_learning_rate"])
        self.max_lr = float(config["training"]["learning_rate"]["max_learning_rate"])
        self.decay_lr = float(config["training"]["learning_rate"]["decay_learning_rate"])

        # Cycle size
        self.cycle_epochs = int(config["training"]["learning_rate"]["cycle_epochs"])
        self.decay_epochs = int(config["training"]["epochs"]) - self.cycle_epochs
        self.start_step = None if config["training"]["learning_rate"].get("hot_start", False) else 0

    def __call__(self, epoch):

        # Update our start step if we haven't run yet
        self.start_step = epoch if self.start_step is None else self.start_step

        # While we are in the one cycle, cycle our learning rate
        cycle_phase = (epoch - self.start_step) / self.cycle_epochs
        if cycle_phase < 0.5:  # Going up
            p = cycle_phase * 2  # Value from 0-1
            return self.min_lr + (self.max_lr - self.min_lr) * p
        elif cycle_phase < 1.0:  # Going down
            p = 1.0 - (cycle_phase - 0.5) * 2  # Value from 0-1
            return self.min_lr + (self.max_lr - self.min_lr) * p

        # After the one cycle, we just decay our learning rate slowly down to nothing
        else:
            decay_phase = (epoch - self.start_step - self.cycle_epochs) / self.decay_epochs
            return self.min_lr * (1.0 - decay_phase) + self.decay_lr * decay_phase

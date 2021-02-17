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

import tensorflow as tf


class ImageTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super(ImageTensorBoard, self).__init__(**kwargs)

    def _image_metrics(self):
        return [m for m in self.model.metrics if hasattr(m, "images")]

    def _filter_logs(self, logs):
        if logs is None:
            return None

        # Find all the metrics that are actually image metrics
        remove_list = self._image_metrics()
        remove_list = [m.name for m in remove_list]
        remove_list.extend(["val_{}".format(n) for n in remove_list])

        return {k: v for k, v in logs.items() if k not in remove_list}

    def on_batch_begin(self, batch, logs=None):
        super(ImageTensorBoard, self).on_batch_begin(batch, self._filter_logs(logs))

    def on_batch_end(self, batch, logs=None):
        super(ImageTensorBoard, self).on_batch_end(batch, self._filter_logs(logs))

    def on_epoch_begin(self, epoch, logs=None):
        super(ImageTensorBoard, self).on_epoch_begin(epoch, self._filter_logs(logs))

    def on_epoch_end(self, epoch, logs=None):

        if logs is not None:
            with self._train_writer.as_default():
                for m in self._image_metrics():
                    tf.summary.image(m.name, m.images(logs[m.name]), epoch)

            if any([l.startswith("val_") for l in logs.keys()]):
                with self._val_writer.as_default():
                    for m in self._image_metrics():
                        tf.summary.image(m.name, m.images(logs["val_{}".format(m.name)]), epoch)

        super(ImageTensorBoard, self).on_epoch_end(epoch, self._filter_logs(logs))

    def on_predict_batch_begin(self, batch, logs=None):
        super(ImageTensorBoard, self).on_predict_batch_begin(batch, self._filter_logs(logs))

    def on_predict_batch_end(self, batch, logs=None):
        super(ImageTensorBoard, self).on_predict_batch_end(batch, self._filter_logs(logs))

    def on_predict_begin(self, logs=None):
        super(ImageTensorBoard, self).on_predict_begin(self._filter_logs(logs))

    def on_predict_end(self, logs=None):
        super(ImageTensorBoard, self).on_predict_end(self._filter_logs(logs))

    def on_test_batch_begin(self, batch, logs=None):
        super(ImageTensorBoard, self).on_test_batch_begin(batch, self._filter_logs(logs))

    def on_test_batch_end(self, batch, logs=None):
        super(ImageTensorBoard, self).on_test_batch_end(batch, self._filter_logs(logs))

    def on_test_begin(self, logs=None):
        super(ImageTensorBoard, self).on_test_begin(self._filter_logs(logs))

    def on_test_end(self, logs=None):
        super(ImageTensorBoard, self).on_test_end(self._filter_logs(logs))

    def on_train_batch_begin(self, batch, logs=None):
        super(ImageTensorBoard, self).on_train_batch_begin(batch, self._filter_logs(logs))

    def on_train_batch_end(self, batch, logs=None):
        super(ImageTensorBoard, self).on_train_batch_end(batch, self._filter_logs(logs))

    def on_train_begin(self, logs=None):
        super(ImageTensorBoard, self).on_train_begin(self._filter_logs(logs))

    def on_train_end(self, logs=None):
        super(ImageTensorBoard, self).on_train_end(self._filter_logs(logs))

    def set_model(self, model):
        super(ImageTensorBoard, self).set_model(model)

    def set_params(self, params):
        super(ImageTensorBoard, self).set_params(params)

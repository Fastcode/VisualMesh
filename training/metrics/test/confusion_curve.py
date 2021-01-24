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

from .curve import Curve


class ConfusionCurve(Curve):
    def __init__(self, class_index, **kwargs):

        super(ConfusionCurve, self).__init__(**kwargs)
        self.class_index = class_index

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Strip out any values that are unlabelled
        idx = tf.squeeze(tf.where(tf.reduce_any(y_pred != 0, axis=-1)), axis=-1)
        y_true = tf.gather(y_true, idx)
        y_pred = tf.gather(y_pred, idx)

        # Count of tp and fp instances and the value predicted
        c = tf.cast(y_true[:, self.class_index], tf.int64)
        c = tf.stack([c, 1 - c], axis=-1)
        X = tf.expand_dims(y_pred[:, self.class_index], axis=-1)
        self.update(X, c)


def _thresholded_confusion(c):
    thresholded = tf.cumsum(c, reverse=True, axis=0)

    rp = tf.reduce_sum(c[:, 0])
    rn = tf.reduce_sum(c[:, 1])

    tp = thresholded[:, 0]
    fp = thresholded[:, 1]
    tn = rn - fp
    fn = rp - tp

    return (tp, fp, tn, fn)


def _tpr(X, c):
    tp, fp, tn, fn = _thresholded_confusion(c)
    v = tp / (tp + fn)
    return tf.where((tp + fn) == 0, tf.ones_like(v), v)


def _tnr(X, c):
    tp, fp, tn, fn = _thresholded_confusion(c)
    v = tn / (tn + fp)
    return tf.where((tn + fp) == 0, tf.ones_like(v), v)


def _ppv(X, c):
    tp, fp, tn, fn = _thresholded_confusion(c)
    v = tp / (tp + fp)
    return tf.where((tp + fp) == 0, tf.ones_like(v), v)


def _npv(X, c):
    tp, fp, tn, fn = _thresholded_confusion(c)
    v = tn / (tn + fn)
    return tf.where((tn + fn) == 0, tf.ones_like(v), v)


def _fnr(X, c):
    tp, fp, tn, fn = _thresholded_confusion(c)
    v = fn / (fn + tp)
    return tf.where((fn + tp) == 0, tf.zeros_like(v), v)


def _fpr(X, c):
    tp, fp, tn, fn = _thresholded_confusion(c)
    v = fp / (fp + tn)
    return tf.where((fp + tn) == 0, tf.zeros_like(v), v)


def _fdr(X, c):
    tp, fp, tn, fn = _thresholded_confusion(c)
    v = fp / (fp + tp)
    return tf.where((fp + tp) == 0, tf.zeros_like(v), v)


def _for(X, c):
    tp, fp, tn, fn = _thresholded_confusion(c)
    v = fn / (fn + tn)
    return tf.where((fn + tn) == 0, tf.zeros_like(v), v)


def _f1(X, c):
    ppv = _ppv(X, c)
    tpr = _tpr(X, c)
    return (ppv * tpr) / (ppv + tpr)


def _informedness(X, c):
    return _tpr(X, c) + _tnr(X, c) - 1.0


def _markedness(X, c):
    return _ppv(X, c) + _npv(X, c) - 1.0


def _mcc(X, c):
    return tf.subtract(
        tf.sqrt(_ppv(X, c) * _tpr(X, c) * _tnr(X, c) * _npv(X, c)),
        tf.sqrt(_fdr(X, c) * _fnr(X, c) * _fpr(X, c) * _for(X, c)),
    )


def _threshold(X, c):
    return tf.cast(tf.squeeze(X, axis=-1), tf.float64)

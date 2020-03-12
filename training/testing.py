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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from . import dataset
from .dataset import VisualMeshDataset
from .model import VisualMeshModel

mpl.use("Agg")


def _thresholded_confusion(X, c):
    thresholded = tf.cumsum(c, reverse=True, axis=0)

    rp = tf.reduce_sum(c[:, 0])
    rn = tf.reduce_sum(c[:, 1])

    tp = thresholded[:, 0]
    fp = thresholded[:, 1]
    tn = rn - fp
    fn = rp - tp

    return (tp, fp, tn, fn)


def _tpr(X, c):
    tp, fp, tn, fn = _thresholded_confusion(X, c)
    return tf.where((tp + fn) == 0, 1.0, tp / (tp + fn))


def _tnr(X, c):
    tp, fp, tn, fn = _thresholded_confusion(X, c)
    return tf.where((tn + fp) == 0, 1.0, tn / (tn + fp))


def _ppv(X, c):
    tp, fp, tn, fn = _thresholded_confusion(X, c)
    return tf.where((tp + fp) == 0, 1.0, tp / (tp + fp))


def _npv(X, c):
    tp, fp, tn, fn = _thresholded_confusion(X, c)
    return tf.where((tn + fn) == 0, 1.0, tn / (tn + fn))


def _fnr(X, c):
    tp, fp, tn, fn = _thresholded_confusion(X, c)
    return tf.where((fn + tp) == 0, 0.0, fn / (fn + tp))


def _fpr(X, c):
    tp, fp, tn, fn = _thresholded_confusion(X, c)
    return tf.where((fp + tn) == 0, 0.0, fp / (fp + tn))


def _fdr(X, c):
    tp, fp, tn, fn = _thresholded_confusion(X, c)
    return tf.where((fp + tp) == 0, 0.0, fp / (fp + tp))


def _for(X, c):
    tp, fp, tn, fn = _thresholded_confusion(X, c)
    return tf.where((fn + tn) == 0, 0.0, fn / (fn + tn))


def _precision(X, c):
    return _ppv(X, c)


def _recall(X, c):
    return _tpr(X, c)


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


def _false_positive_rate(X, c):
    tp, fp, tn, fn = _thresholded_confusion(X, c)
    return fp / (fp + tn)


def _threshold(X, c):
    return tf.cast(X, tf.float64)


def _reduce_curve(X, c, n_bins, x_func, y_func):

    if tf.size(X) < n_bins:
        return (X, c)

    # Sort by confidence
    idx = tf.argsort(X)
    X = tf.gather(X, idx)
    c = tf.gather(c, idx)

    # Calculate the x and y value for all the thresholds we have
    x_values = x_func(X, c)
    y_values = y_func(X, c)

    # Calculate the distance along the PR curve for each point so we can sample evenly along the PR curve
    curve = tf.math.sqrt(
        tf.add(
            tf.math.squared_difference(x_values[:-1], x_values[1:]),
            tf.math.squared_difference(y_values[:-1], y_values[1:]),
        )
    )
    curve = tf.pad(tf.cumsum(curve), [[1, 0]])  # First point is 0 length along the curve

    # Use cumsum and scatter to try to distribute points as evenly along the PR curve as we can
    idx = tf.cast(tf.expand_dims(tf.multiply(curve, tf.math.divide(n_bins, curve[-1])), axis=-1), dtype=tf.int32)
    idx = tf.clip_by_value(idx, 0, n_bins - 1)
    values = tf.math.reduce_sum(c, axis=-1)
    h_X = tf.scatter_nd(idx, tf.multiply(X, tf.cast(values, X.dtype)), [n_bins])  # Scale by values in bin
    h_c = tf.scatter_nd(idx, c, [n_bins, 2])
    h_X = tf.math.divide(h_X, tf.cast(tf.reduce_sum(h_c, axis=-1), h_X.dtype))  # Renormalise by values in bin

    # Remove any points from the histogram that didn't end up getting values
    idx = tf.squeeze(tf.where(tf.logical_not(tf.reduce_all(h_c == 0, axis=-1))), axis=-1)
    h_X = tf.gather(h_X, idx)
    h_c = tf.gather(h_c, idx)

    return h_X, h_c


def test(config, output_path):

    classes = config["network"]["classes"]
    n_classes = len(classes)
    n_bins = config["testing"]["n_bins"]

    # Find the latest checkpoint file and load it
    checkpoint_file = tf.train.latest_checkpoint(output_path)
    if checkpoint_file is not None:
        model = tf.keras.models.load_model(checkpoint_file)
    else:
        raise RuntimeError("There was no checkpoint to load in the output_path")

    # Count how many elements there are in the dataset
    n_records = sum(
        1
        for _ in tqdm(
            tf.data.TFRecordDataset(config["dataset"]["testing"]),
            dynamic_ncols=True,
            leave=False,
            unit=" records",
            desc="Counting records in test dataset",
        )
    )
    n_batches = n_records // max(1, config["testing"]["batch_size"])
    print("Testing using {} records in {} batches".format(n_records, n_batches))

    # Load the testing dataset
    validation_dataset = VisualMeshDataset(
        input_files=config["dataset"]["testing"],
        classes=classes,
        mesh=config["model"]["mesh"],
        geometry=config["model"]["geometry"],
        batch_size=config["testing"]["batch_size"],
        prefetch=tf.data.experimental.AUTOTUNE,
        variants={},
    ).build()

    curve_types = [
        ("pr", _recall, _precision),
        ("roc", _false_positive_rate, _recall),
        ("mi", _informedness, _markedness),
        ("precision", _threshold, _precision),
        ("recall", _threshold, _recall),
        ("f1", _threshold, _f1),
        ("informedness", _threshold, _informedness),
        ("markedness", _threshold, _markedness),
        ("mcc", _threshold, _mcc),
    ]

    # Storage for the metric information
    confusion = tf.Variable(tf.zeros(shape=(n_classes, n_classes), dtype=tf.int64))
    curves = [{c: [] for c, x, y in curve_types} for i in range(n_classes)]

    for e in tqdm(validation_dataset, dynamic_ncols=True, unit=" batches", leave=True, total=n_batches):
        # Run the predictions
        X = model.predict_on_batch((e["X"], e["G"]))
        Y = e["Y"]

        # Strip down all the elements that don't have a label
        idx = tf.squeeze(tf.where(tf.reduce_any(tf.greater(Y, 0.0), axis=-1)), axis=-1)
        X = tf.gather(X, idx)
        Y = tf.gather(Y, idx)

        # Build up an index list that maps each class and add them into the corresponding locations
        idx = tf.stack([tf.argmax(input=Y, axis=-1), tf.argmax(input=X, axis=-1)], axis=-1)
        confusion.assign_add(tf.scatter_nd(idx, tf.ones_like(idx[:, 0], dtype=confusion.dtype), confusion.shape))

        # Build the threshold curves for each class
        tp = tf.cast(Y, tf.int64)
        fp = 1 - tp
        for i in range(n_classes):
            # Extract the relevant information
            X_c = X[:, i]
            tp_c = tp[:, i]
            fp_c = fp[:, i]
            tpfp = tf.stack([tp_c, fp_c], axis=-1)

            # Add our reduced curve
            for c, x_func, y_func in curve_types:
                curves[i][c].append(_reduce_curve(X_c, tpfp, n_bins, x_func, y_func))

    # Go through our output and calculate the metrics we can
    metrics = []
    for i in range(n_classes):

        # Calculate interclass precision and recall
        tp_fn = tf.reduce_sum(confusion[i, :])
        p = tf.reduce_sum(confusion[:, i])
        class_recall = tf.stack([confusion[i, j] / tp_fn for j in range(n_classes)])
        class_precision = tf.stack([confusion[j, i] / p for j in range(n_classes)])

        # Reduce the curves down
        for name, x_func, y_func in curve_types:

            # Concatentate all our data and reduce the curve
            data = curves[i][name]

            c = _reduce_curve(
                tf.concat([v[0] for v in data], axis=0),
                tf.concat([v[1] for v in data], axis=0),
                n_bins,
                x_func,
                y_func,
            )

            # Calculate our x, y and threshold for this point and overwrite the plot
            curves[i][name] = (x_func(*c), y_func(*c), _threshold(*c))

        metrics.append(
            {
                "curves": curves[i],
                "precision": class_precision,
                "recall": class_recall,
                # Negative here since recall goes down as threshold goes up
                "ap": -np.trapz(curves[i]["pr"][1].numpy(), curves[i]["pr"][0].numpy()),
            }
        )

    # Write out the mAP and ap for each class
    os.makedirs(os.path.join(output_path, "test"), exist_ok=True)
    with open(os.path.join(output_path, "test", "pr.txt"), "w") as f:

        def write(txt):
            print("{}".format(txt))
            f.write("{}\n".format(txt))

        # Global metrics
        write("Global:")
        write("\tmAP: {}".format(sum([m["ap"] for m in metrics]) / n_classes))
        write("\tAverage Precision: {}".format(sum([m["precision"][i] for i, m in enumerate(metrics)]) / n_classes))
        write("\tAverage Recall: {}".format(sum([m["recall"][i] for i, m in enumerate(metrics)]) / n_classes))
        write("")

        # Individual precisions
        for i, m in enumerate(metrics):

            # Save the metrics
            name = classes[i]["name"]
            write("{}".format(name.title()))
            write("\tAP: {}".format(m["ap"]))
            write("\tPrecision: {}".format(m["precision"][i]))
            write("\tRecall: {}".format(m["recall"][i]))
            write("\tPredicted {} samples are really:".format(name.title()))
            for j in range(n_classes):
                write("\t\t{}: {:.3f}%".format(classes[j]["name"].title(), 100 * m["precision"][j]))
            write("\tReal {} samples are predicted as:".format(name.title()))
            for j in range(n_classes):
                write("\t\t{}: {:.3f}%".format(classes[j]["name"].title(), 100 * m["recall"][j]))

            for curve_name, x_func, y_func in curve_types:
                # Save the curves
                np.savetxt(
                    os.path.join(output_path, "test", "{}_{}.csv".format(name, curve_name)),
                    tf.stack(m["curves"][curve_name], axis=-1).numpy(),
                    comments="",
                    header="{},{},Threshold".format(x_func.__name__[1:].title(), y_func.__name__[1:].title()),
                    delimiter=",",
                )

            max_f1 = m["curves"]["f1"][0][tf.argmax(m["curves"]["f1"][1])]
            max_informedness = m["curves"]["informedness"][0][tf.argmax(m["curves"]["informedness"][1])]
            max_markedness = m["curves"]["markedness"][0][tf.argmax(m["curves"]["markedness"][1])]
            max_mcc = m["curves"]["mcc"][0][tf.argmax(m["curves"]["mcc"][1])]

            def matching_point(curve, threshold):
                upper = min(tf.size(curve[2]) - 1, np.searchsorted(curve[2], threshold))
                lower = max(0, upper - 1)

                min_t = curve[2][lower]
                max_t = curve[2][upper]
                factor = (threshold - min_t) / (max_t - min_t)

                return (
                    [(curve[0][lower] * (1.0 - factor) + curve[0][upper] * factor).numpy()],
                    [(curve[1][lower] * (1.0 - factor) + curve[1][upper] * factor).numpy()],
                )

            # Precision recall plot
            curve = m["curves"]["pr"]
            fig, ax = plt.subplots()
            ax.plot(curve[0], curve[1])
            ax.scatter(*matching_point(curve, max_f1)).set_label(r"$f_{1max}$")
            ax.scatter(*matching_point(curve, max_informedness)).set_label(r"$informedness_{max}$")
            ax.scatter(*matching_point(curve, max_markedness)).set_label(r"$markedness_{max}$")
            ax.scatter(*matching_point(curve, max_mcc)).set_label(r"$\phi_{max}$")
            ax.legend()
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("{} Precision/Recall".format(name.title()))
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            fig.savefig(os.path.join(output_path, "test", "{}_pr.pdf".format(name)))
            plt.close(fig)

            # ROC plot
            curve = m["curves"]["roc"]
            fig, ax = plt.subplots()
            ax.plot(curve[0], curve[1])
            ax.scatter(*matching_point(curve, max_f1)).set_label(r"$f_{1max}$")
            ax.scatter(*matching_point(curve, max_informedness)).set_label(r"$informedness_{max}$")
            ax.scatter(*matching_point(curve, max_markedness)).set_label(r"$markedness_{max}$")
            ax.scatter(*matching_point(curve, max_mcc)).set_label(r"$\phi_{max}$")
            ax.legend()
            ax.set_xlabel("False Postive Rate")
            ax.set_ylabel("True Postive Rate")
            ax.set_title("{} ROC".format(name.title()))
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            fig.savefig(os.path.join(output_path, "test", "{}_roc.pdf".format(name)))
            plt.close(fig)

            # Markedness Informedness plot
            curve = m["curves"]["mi"]
            fig, ax = plt.subplots()
            ax.plot(curve[0], curve[1])
            ax.scatter(*matching_point(curve, max_f1)).set_label(r"$f_{1max}$")
            ax.scatter(*matching_point(curve, max_mcc)).set_label(r"$\phi_{max}$")
            ax.legend()
            ax.set_xlabel("Markedness")
            ax.set_ylabel("Informedness")
            ax.set_title("{} Markedness/Informedness".format(name.title()))
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            fig.savefig(os.path.join(output_path, "test", "{}_mi.pdf".format(name)))
            plt.close(fig)

            # Mixed information plot
            fig, ax = plt.subplots()
            ax.plot(m["curves"]["precision"][0], m["curves"]["precision"][1])[0].set_label(r"$Precision$")
            ax.plot(m["curves"]["recall"][0], m["curves"]["recall"][1])[0].set_label(r"$Recall$")
            ax.plot(m["curves"]["f1"][0], m["curves"]["f1"][1])[0].set_label(r"$f_{1}$")
            ax.plot(m["curves"]["informedness"][0], m["curves"]["informedness"][1])[0].set_label(r"$Informedness$")
            ax.plot(m["curves"]["markedness"][0], m["curves"]["markedness"][1])[0].set_label(r"$Markedness$")
            ax.plot(m["curves"]["mcc"][0], m["curves"]["mcc"][1])[0].set_label(r"$\phi Coefficent$")
            ax.legend()
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Value")
            ax.set_title("{} Metrics".format(name.title()))
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            fig.savefig(os.path.join(output_path, "test", "{}_metric.pdf".format(name)))
            plt.close(fig)

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

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from . import dataset
from .dataset import VisualMeshDataset
from .model import VisualMeshModel


def _precision(X, c):
    thresholded = tf.cumsum(c, reverse=True, axis=0)
    return tf.math.divide(thresholded[:, 0], tf.add(thresholded[:, 0], thresholded[:, 1]))


def _recall(X, c):
    thresholded = tf.cumsum(c, reverse=True, axis=0)
    p = tf.reduce_sum(c[:, 0], axis=0)
    return tf.math.divide(thresholded[:, 0], p)


def _threshold(X, c):
    return tf.cast(X, tf.float64)


def _reduce_curve(X, c, n_bins, x_axis, y_axis):

    # Sort by confidence
    idx = tf.argsort(X)
    X = tf.gather(X, idx)
    c = tf.gather(c, idx)

    # Calculate the x and y value for all the thresholds we have
    x_values = x_axis(X, c)
    y_values = y_axis(X, c)

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

    # If we don't have enough values to make up the bins, just return the values themselves
    X = tf.cond(tf.size(X) < n_bins, lambda: X, lambda: h_X)
    c = tf.cond(tf.size(c) < n_bins, lambda: c, lambda: h_c)

    return X, c


def test(config, output_path):

    classes = config.network.classes
    n_classes = len(classes)
    n_bins = config.testing.n_bins

    # Define the model
    model = VisualMeshModel(
        structure=config.network.structure, n_classes=n_classes, activation=config.network.activation_fn,
    )

    # Find the latest checkpoint file and load it
    checkpoint_file = tf.train.latest_checkpoint(output_path)
    if checkpoint_file is not None:
        model.load_weights(checkpoint_file)
    else:
        raise RuntimeError("There was no checkpoint to load in the output_path")

    # Count how many elements there are in the dataset
    n_records = sum(
        1
        for _ in tqdm(
            tf.data.TFRecordDataset(config.dataset.testing),
            dynamic_ncols=True,
            leave=False,
            unit=" records",
            desc="Counting records in test dataset",
        )
    )
    n_batches = n_records // max(1, config.testing.batch_size)
    print("Testing using {} records in {} batches".format(n_records, n_batches))

    # Load the testing dataset
    validation_dataset = VisualMeshDataset(
        input_files=config.dataset.testing,
        classes=classes,
        model=config.model,
        batch_size=config.testing.batch_size,
        prefetch=tf.data.experimental.AUTOTUNE,
        variants={},
    ).build()

    # Storage for the metric information
    confusion = tf.Variable(tf.zeros(shape=(n_classes, n_classes), dtype=tf.int64))
    curves = [{"recall_threshold": [], "precision_threshold": [], "precision_recall": []} for i in range(n_classes)]

    v = 0
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
        for i, c in enumerate(classes):
            # Extract the relevant information
            X_c = X[:, i]
            tp_c = tp[:, i]
            fp_c = fp[:, i]
            tpfp = tf.stack([tp_c, fp_c], axis=-1)

            curves[i]["precision_recall"].append(_reduce_curve(X_c, tpfp, n_bins, _recall, _precision))
            curves[i]["precision_threshold"].append(_reduce_curve(X_c, tpfp, n_bins, _threshold, _precision))
            curves[i]["recall_threshold"].append(_reduce_curve(X_c, tpfp, n_bins, _threshold, _recall))

        v = v + 1
        if v == 20:
            break

    # Go through our output and calculate the metrics we can
    metrics = []
    for i, c in enumerate(classes):

        # Calculate interclass precision and recall
        tp_fn = tf.reduce_sum(confusion[i, :])
        p = tf.reduce_sum(confusion[:, i])
        class_recall = tf.stack([confusion[i, j] / tp_fn for j in range(n_classes)])
        class_precision = tf.stack([confusion[j, i] / p for j in range(n_classes)])

        # Extract the curve information
        pr = curves[i]["precision_recall"]
        pr = _reduce_curve(
            tf.concat([v[0] for v in pr], axis=0), tf.concat([v[1] for v in pr], axis=0), n_bins, _recall, _precision
        )

        # For the precision recall curve specifically we need to add on a first point
        # and a last point if we didn't reach the end of the curve
        all_points_precision = tf.reduce_sum(pr[1][:, 0]) / tf.reduce_sum(pr[1])
        pr = (
            tf.concat([_recall(*pr), [0.0]], axis=0),
            tf.concat([_precision(*pr), [1.0]], axis=0),
            tf.concat([_threshold(*pr), [1.0]], axis=0),
        )
        if tf.reduce_max(pr[0]) < 1.0:
            pr = (
                tf.concat([pr[0], [1.0]], axis=0),
                tf.concat([pr[1], [all_points_precision]], axis=0),
                tf.concat([pr[2], [0.0]], axis=0),
            )

        idx = tf.argsort(pr[0], stable=True)
        pr = tuple(tf.gather(p, idx) for p in pr)

        pt = curves[i]["precision_threshold"]
        pt = _reduce_curve(
            tf.concat([v[0] for v in pt], axis=0),
            tf.concat([v[1] for v in pt], axis=0),
            n_bins,
            _threshold,
            _precision,
        )
        pt = (_threshold(*pt), _precision(*pt))

        rt = curves[i]["recall_threshold"]
        rt = _reduce_curve(
            tf.concat([v[0] for v in rt], axis=0), tf.concat([v[1] for v in rt], axis=0), n_bins, _threshold, _recall,
        )
        rt = (_threshold(*rt), _recall(*rt))

        metrics.append(
            {
                "curves": {"precision_recall": pr, "precision_threshold": pt, "recall_threshold": rt},
                "precision": class_precision,
                "recall": class_recall,
                "ap": np.trapz(pr[1], pr[0]),
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
            name = classes[i][0]
            write("{}".format(name.title()))
            write("\tAP: {}".format(m["ap"]))
            write("\tPrecision: {}".format(m["precision"][i]))
            write("\tRecall: {}".format(m["recall"][i]))
            write(
                "\tClass Precision: [{}]".format(
                    ", ".join(["{}:{}".format(classes[j][0], m["precision"][j]) for j in range(n_classes)])
                )
            )
            write(
                "\tClass Recall: [{}]".format(
                    ", ".join(["{}:{}".format(classes[j][0], m["recall"][j]) for j in range(n_classes)])
                )
            )

            # Save the curves
            np.savetxt(
                os.path.join(output_path, "test", "{}_precision_recall.csv".format(name)),
                tf.stack(m["curves"]["precision_recall"], axis=-1).numpy(),
                comments="",
                header="Recall,Precision,Threshold",
                delimiter=",",
            )
            np.savetxt(
                os.path.join(output_path, "test", "{}_precision_threshold.csv".format(name)),
                tf.stack(m["curves"]["precision_threshold"], axis=-1).numpy(),
                comments="",
                header="Threshold,Precision",
                delimiter=",",
            )
            np.savetxt(
                os.path.join(output_path, "test", "{}_recall_threshold.csv".format(name)),
                tf.stack(m["curves"]["recall_threshold"], axis=-1).numpy(),
                comments="",
                header="Threshold,Recall",
                delimiter=",",
            )

            # Make curve images using matplotlib

    import pdb

    pdb.set_trace()

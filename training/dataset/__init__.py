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

from .example import Image
from .label import Classification, Seeker
from .orientation import Ground, Spotlight
from .projection import VisualMesh
from .view import Monoscopic, Stereoscopic
from .visual_mesh_dataset import VisualMeshDataset


def Dataset(paths, view, example, orientation, label, projection, keys):

    # Find the correct class to handle our view
    if view["type"] == "Monoscopic":
        view = Monoscopic(**view["config"])
    elif view["type"] == "Stereoscopic":
        view = Stereoscopic(**view["config"])
    else:
        raise RuntimeError("Unknown view type '{}'".format(view["type"]))

    # Find the correct class to handle our view
    if example["type"] == "Image":
        example = Image(**example["config"])
    else:
        raise RuntimeError("Unknown example type '{}'".format(example["type"]))

    # Find the correct class to handle our mesh orientation
    if orientation["type"] == "Ground":
        orientation = Ground(**orientation["config"])
    elif orientation["type"] == "Spotlight":
        orientation = Spotlight(**orientation["config"])
    else:
        raise RuntimeError("Unknown orientation type '{}'".format(orientation["type"]))

    # Find the correct class to handle our mesh orientation
    if projection["type"] == "VisualMesh":
        projection = VisualMesh(**projection["config"])
    else:
        raise RuntimeError("Unknown projection type '{}'".format(projection["type"]))

    # Find the correct dataset labelling class
    if label["type"] == "Classification":
        label = Classification(**label["config"])
    elif label["type"] == "Seeker":
        label = Seeker(**label["config"])
    else:
        raise RuntimeError("Unknown data labelling scheme '{}'".format(label["type"]))

    # Create our dataset with these specific components
    return VisualMeshDataset(
        paths=paths, view=view, example=example, orientation=orientation, projection=projection, label=label, keys=keys,
    ).build()


# Convert a dataset into a format that will be accepted by keras fit
def keras_dataset(args):
    # Return in the format (x, y, weights)
    return ((args["X"], args["G"]), args["Y"])

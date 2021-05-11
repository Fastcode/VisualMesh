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

from ..callbacks import ClassificationImages, SeekerImages
from .dataset import Dataset
from .merge_configuration import merge_configuration


def ImageCallback(config, output_path):

    validation_config = merge_configuration(config, config["dataset"].get("config", {}))

    if config["label"]["type"] == "Classification":

        n_images = config["training"]["validation"]["progress_images"]
        classes = config["label"]["config"]["classes"]

        return ClassificationImages(
            output_path=output_path,
            dataset=Dataset(config, "validation", batch_size=n_images).take(1),
            # Draw using the first colour in the list for each class
            colours=[c["colours"][0] for c in classes],
        )

    elif config["label"]["type"] == "Seeker":

        n_images = config["training"]["validation"]["progress_images"]

        return SeekerImages(
            output_path=output_path,
            dataset=Dataset(config, "validation", batch_size=n_images).take(1),
            model=validation_config["projection"]["config"]["mesh"]["model"],
            max_distance=validation_config["projection"]["config"]["mesh"]["max_distance"],
            geometry=validation_config["projection"]["config"]["geometry"]["shape"],
            radius=validation_config["projection"]["config"]["geometry"]["radius"],
            scale=validation_config["label"]["config"]["scale"],
        )

    else:
        raise RuntimeError("Cannot create images callback, {} is not a supported type".format(config["label"]["type"]))

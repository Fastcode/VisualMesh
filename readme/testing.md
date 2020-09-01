# Testing
Testing in the Visual Mesh code is run after training in a similar way to training.

For example run the following using docker
```sh
./docker run --gpus all -u $(id -u):$(id -g) -it --rm --volume $(pwd):/workspace visualmesh:latest ./mesh.py test <path/to/output>
```

The output of the tests will be placed into the output directory.
This includes several graph images as well as some statistics.

Due to the way these graphs are produced they will be much more accurate than if a simple histogram were used and can be used to make much more complex shapes.
However they unfortunately can suffer from an issue at the ends of the plot.
The easiest way to fix this is to delete the start/end of the data in the CSV file to remove the erroneous sections.

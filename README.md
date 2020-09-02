# Visual Mesh 2
[![Join the chat at https://gitter.im/Fastcode/VisualMesh](https://badges.gitter.im/Fastcode/VisualMesh.svg)](https://gitter.im/Fastcode/VisualMesh?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
If you have any questions or discussions while using the visual mesh, feel free to ask on gitter.

The Visual Mesh is an input transformation that uses knowledge of a cameras orientation and position relative to an observation plane to greatly increase the performance and accuracy of a convolutional neural network.
It utilises the geometry of objects to create a mesh structure that ensures that a similar number of samples points are selected regardless of distance.
The result is that networks can be much smaller and simpler while still achieving high levels of accuracy.
Additionally it is very capable when it comes to detecting distant objects.
Normally distant objects become too small for a network to detect accurately, but as the visual mesh normalises its size detections are still accurate.

 | | |
 |:-:|:-:|
 |![](readme/distant.png)|![](readme/mesh.jpg)|
 | | |

This codebase provides two components, one is a TensorFlow 2 based training and testing system that allows you to build and train networks.
The other component is a C++ API that is designed to operate as a high performance inference engine using OpenCL.
This allows you to get high performance on systems with devices like Intel integrated GPUs with hundreds of frames per second.

## [Quick Start Guide](readme/quickstart.md)
If all you want to do is train a simple classification network for your own dataset take a look at the [Quick Start Guide](readme/quickstart.md)

## [Setup](readme/setup.md)
For setup, you can either build the custom training op using CMake, or you can build a docker image to train with via docker.
You can get a more detailed description on the [Setup Page](readme/setup.md).

## [Architecture](readme/architecture.md)
The architecture of the training system is based on the concept of combining several different concepts into a network.
These concepts are called flavours and define the types of inputs and outputs that the network handles as well as details such as the loss function.
Find out more in the [Architecture Page](readme/architecture.md).

## [Dataset](readme/dataset.md)
The dataset for the visual mesh is provided via TFRecord files.
The keys that are used in the TFRecord files are determined by the flavour that is used.
See the [Dataset Page](readme/dataset.md) for more information.

## [Training](readme/training.md)
In order to train a network using the visual mesh, you need to place a configuration file in your output folder.
Then you are able to run the following command to train the network.
```sh
./mesh.py train <path/to/output>
```
Or using docker:
```sh
./docker run --gpus all -u $(id -u):$(id -g) -it --rm --volume $(pwd):/workspace visualmesh:latest ./mesh.py train <path/to/output>
```
More information about the training process as well as the influence of different parameters in the configuration can be found on the **[Training Page](readme/training.md)**

## [Testing](readme/testing.md)
This codebase provides a suite of testing tools in order to determine the efficacy of a trained network.
It outputs metrics (e.g. precision and recall) and has a system for drawing charts from combinations of properties.
More information about the testing and the graphs that it outputs can be found at the **[Testing Page](readme/testing.md)**

## [Inference](readme/inference.md)
The visual mesh codebase also provides a custom C++ inference engine that is able to be executed on a wide variety of devices.
The inference code is based on several different engines.
Each of these engines has a different requirement in order to build code using it.
For example, the OpenCL engine requires that you have an OpenCL implementation and appropriate drivers installed when you build and run the code.
If you don't have these the code will still work, however, you will not have access to those engines.
For more information see the **[Inference Page](readme/inference.md)**

## Contributing
Contributions to this project are welcome via pull request.
Style guides are enforced by clang-format for C++ and black for Python.

## Citing
If you use the VisualMesh in your work, please cite it
```
Houliston T., Chalup S.K. (2019) Visual Mesh: Real-Time Object Detection Using Constant Sample Density. In: Holz D., Genter K., Saad M., von Stryk O. (eds) RoboCup 2018: Robot World Cup XXII. RoboCup 2018. Lecture Notes in Computer Science, vol 11374. Springer, Cham
```

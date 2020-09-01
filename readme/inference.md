# Inference
The c++ inference engine is split into two main components, the mesh components and the engine components.
The mesh components are for building up a visual mesh object based on the position of the observation plane relative to the camera.
The engines take these mesh objects and run projection and inference code on them.

## Exporting to YAML
If you have trained a network and wish to output it the a yaml format that most of the examples use you can use the export function.
```sh
./mesh.py export <path/to/output>
```
This will create a yaml file with the weights and network in it ready for use.

## Mesh
The mesh objects generate a single look up table of the entire graph.
The mesh objects are able to look up which of the points in the visual mesh are on screen.
They do this using a binary search partition tree to quickly narrow down which points are valid.
However, since this tree isn't perfect you may occasionally get points that are slightly off screen.
If it is important that you do not have any points off screen you should add some code to check for this.

There are two main mesh objects that are available in the visual mesh codebase.
The first is the `visualmesh::Mesh` class.
This class holds a single visual mesh for a specific height.
The second is the `visualmesh::VisualMesh` which holds multiple different `visualmesh::Mesh` objects for different heights that ensure that the error in number of intersections does not grow beyond a target value.
They both can be used with engines in the same way, and if using `visualmesh::VisualMesh` it will select the closest matching height for the Hoc used.

It is created via a template `visualmesh::Mesh<Scalar, Model>` where the Scalar is the datatype that the mesh will be created with (for example `float` or `double`).
The model is the specific visual mesh generation model that will be used.
For example `visualmesh::model::Ring6` would select the six neighbour ring model.
You can find more about the models that are available and their advantages and disadvantages [here](flavour/projection.md).
Make sure that you use the same model that you trained the network for.
It is often advantageous to create the Mesh object using double precision, and then convert it to single precision float.
This results in a higher accuracy network especially as distances increase.
```cpp
visualmesh::Mesh<float, visualmesh::model::Ring6> mesh = visualmesh::Mesh<double, visualmesh::model::Ring6>(visualmesh::geometry::Sphere<double>(0.05), 1.0, 5, 20);
```

## Engines
The engines are the parts of the code that do the heavy lifting of classification and projection for the codebase.
They are created with neural network weights and will build the network to be executed internally.
They are then able to be used with a mesh and camera object to perform the visual mesh operation.

Once created you use them calling the objects with a mesh and lens information.
```cpp
visualmesh::Mesh<float, visualmesh::model::Ring6> mesh(visualmesh::geometry::Sphere<double>(0.05), 1.0, 5, 20);
visualmesh::engine::cpu::Engine<Scalar> engine(network);

engine(mesh, Hoc, image, format);
```

The engines that are currently available in the system are:

### CPU Engine
This engine is designed to be a reference implementation for the visual mesh.
It is not the fastest engine and does not take advantage of multithreading or other devices.
Use this engine if you don't care about performance and just want to test networks

### OpenCL Engine
This engine generates OpenCL kernels on the fly which it uses to run the inference.
You can use this engine to run on a wide variety of CPU and GPU hardware and it is high performance.

### Vulkan Engine (incomplete)
The vulkan engine is based on the Vulkan GPU api.
It is not yet complete and will occasionally cause your entire computer to freeze up and become unresponsive.
Use at your own risk.

### Future Engines
In the future there are plans to implement a TensorRT engine and a CUDA engine.
Pull request welcome!

## Multithreading
**The engine instances are not thread safe!**
Each of the engine instances are designed not to be thread safe to allow for maximum performance.
When executing the network, you will find the utilisation of your platform is low as quite a bit of time is spent enqueueing kernels and uploading/downloading data from devices.
Because of this you need to interleave your requests on the device.
To do this make multiple engine instances for your same network and ensure that only a single thread is using one at a time.
This allows multiple threads to be enqueuing/running data ont he device at the same time and will greatly improve your performance.
For an example of this you can look at the `benchmark.cpp` example code which uses this principle to achieve higher framerates.

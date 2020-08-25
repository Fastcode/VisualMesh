# Flavours
The Visual Mesh code is designed to easily be able to implement new ideas for how to use it.
It allows this by providing several different components that can be combined to make a vision system.
In the code these components are referred to as flavours.
There are five different categories for these that are each discussed on their own pages.

[View](readme/flavour/view.md) describes how the system is set up in terms of cameras.
It can be used to implement features like Stereoscopic or Multi Camera systems.

[Orientation](readme/flavour/orientation.md) describes how the visual mesh's observation plane is set up from the perspective of the camera.
For example the traditional position for the observation plane is on the ground below the camera.
However if you were to attach the plane to another flat surface this flavour can provide the tools to do that.

[Projection](readme/flavour/projection.md) describes how the visual mesh itself is designed and projected into pixel coordinates.
It includes details about which visual mesh model to use and also contains the logic that uses lens parameters to project to image space.

[Example](readme/flavour/example.md) describes how the training input examples are provided.
For example, the Image method will load images from the dataset and grab the pixels projected by the visual mesh.
If you find some other wonderful way to use the visual mesh besides images put it here!

[Label](readme/flavour/label.md) describes how the training labels are provided.
For a classification network this would be the mask image that the labels are gathered from.

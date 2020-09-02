# Orientation
A systems orientation describes where the observation plane is as related to the camera.
The expected output from this flavour is a key `Hoc: float[4, 4]`.
This key is a homogenous transformation matrix which if used like `Hoc * (a,b,c,1)` would convert a vector in the cameras coordinate system into one measured in the observation planes coordinate system.
In this matrix, the visual mesh only cares about rotation and the z component of the translation for height.
```
┌            ┐
│          0 │
│    R     0 │
│          z │ <- Only Z is important for the visual mesh
│ 0  0  0  1 │
└            ┘
```

For the purposes of the visual mesh, the cameras coordinate system is defined as a right-handed coordinate system with the X axis travelling along the optical axis (direction of view) the Y axis pointing to the left of the camera, and the Z axis pointing upward from the camera.

## Ground
The ground orientation is the most typical use case for the visual mesh.
In this set up the Hoc provided by the dataset is forwarded on as `Hoc` for the output.

### Dataset Keys
The only dataset key required by the Ground flavour is `Hoc` which will be forwarded on as Hoc for the network.
```python
"Hoc": float[4, 4]
```

### Configuration
```yaml
orientation:
  type: Ground
  config:
    augmentations:
      # Adjust the height above the observation plane.
      height: { mean: 0, stddev: 0.05 }
      # Rotate around a random axis by this angle
      rotation: { mean: 0, stddev: 0.0872665 }
```

## Spotlight
The spotlight mesh is based on the idea of projecting a visual mesh plane at some specific point relative to the camera.
It is given a list of targets and will choose one of these targets at random.
When training over multiple epochs this random selection is done each time the example is shown.
This object is projected and used as the origin for a visual mesh observation plane that is tangential to the location of the object.
When this projection is done, the X-axis of the visual mesh will always be aligned with the Z-axis of the observation plane.
This ensures consistent results regardless of where the target is looking

### Dataset Keys
The spotlight network requires a `Hoc` which describes the orientation of the camera relative to the observation plane, the same as in Ground.
It additionally requires targets to points that the mesh will be projected to.
The targets are 3D vectors measured in the cameras coordinate system.
```python
"Hoc": float[4, 4]
"spotlight/targets": float[n, 3] # n >= 1
```

### Configuration
```yaml
orientation:
  type: Spotlight
  config:
    augmentations:
      # Rotate around a random axis by this angle
      rotation: { mean: 0, stddev: 0.0872665 }
      # Adjust the position of the spotlight target
      # Make sure to use min and max to limit objects min distance
      # Otherwise you can end up with an invalid mesh
      # Typically set it to the diameter of the object you are looking for
      position: { mean: 0, stddev: 0.5, min: 0.5, max: 50.0 }
```

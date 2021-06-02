# Dataset
The dataset for the visual mesh should be provided as at least three TFRecord files, these being one or more for each of training, validation and testing.

Each of the flavours used in the network will add requirements for the data that needs to be provided.
You can look these up in each of the individual flavours and how they influence the dataset on the [Architecture Page](./architecture.md).
For example, if you were to make a dataset for a Monoscopic Image Ground Classification Mesh it would need to have the following keys in it:

```python
"Hoc": float[4, 4] # 3D Affine transformation from camera space to observation space
"image": bytes[1] # Compressed image
"mask": bytes[1] # png image
"lens/projection": bytes[1] # 'RECTILINEAR' | 'EQUISOLID' | 'EQUIDISTANT'
"lens/focal_length": float[1] # pixels
"lens/fov": float[1] # radians
"lens/centre": float[2] # pixels
"lens/k": float[2] # pixels
```

If you have a TFRecord dataset which has the correct data types in it, but the keys are incorrect you are able to use the keys field in the configuration file in order to map the keys across.
For example, if you had a field in your dataset `camera/field_of_view` which contained the data required by `lens/fov` you would be able to set up your configuration file like so.
```yaml
dataset:
  training:
    paths:
      - dataset/training.tfrecord
    keys:
      lens/fov: camera/field_of_view
```
This would allow the dataset to look for the data needed by `lens/fov` in the field `camera/field_of_view`

You are also able to override any of the settings for the flavours for only a specific dataset using a config tag.
This is typically used if you want to provide augmentations to your training data, but not apply them when validating or testing the network.

For example, to manipulate the ground plane from the Ground orientation flavour only in training:

```yaml
dataset:
  training:
    paths:
      - dataset/training.tfrecord
    config:
      orientation:
        augmentations:
          height: { mean: 0, stddev: 0.05 }
          rotation: { mean: 0, stddev: 0.0872665 }
```

To make a simple dataset you can follow the instructions in [Quick Start Guide](quickstart.md) or read the code in [training/dataset.py](../training/make_dataset.py)

## Batching
For the visual mesh, the method used for batching is very different from how most batching systems work.
Because of the projection, each image that is loaded into the network will likely have a different number of pixels that are projected onto the image.
This results in a variable-length that would go into the batch.
Therefore instead of adding a new dimension and batching over that dimension we instead concatenate all the samples together.
Then to fix the network we update their graph indices by offsetting them based on their position in the concatenation.
This way the graph will ensure that the ragged length batches will continue to interact properly once concatenated.

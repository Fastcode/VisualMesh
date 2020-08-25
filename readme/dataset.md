# Dataset
The dataset for the visual mesh should be provided as at least three TFRecord files, these being one or more for each of training, validation and testing.

Each of the flavours used in the network will add requirements for the data that needs to be provided.
You can look these up in each of the individual flavour and how the influence the dataset on the [Architecture Page](./architecture.md).
For example if you were to make a dataset for a Monoscopic Image Ground Classification Mesh it would need to have the following keys in it:

```python
"Hoc": float[4, 4]
"image": bytes[1] # Compressed image
"mask": bytes[1] # png image
"lens/projection": bytes[1] # 'RECTILINEAR' | 'EQUISOLID' | 'EQUIDISTANT'
"lens/focal_length": float[1]
"lens/fov": float[1]
"lens/centre": float[2]
"lens/k": float[2]
```

If you have a TFRecord dataset which has the correct data types in it, but the keys are incorrect you are able to use the keys field in order to map the keys across.
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
This is typically used if you want to provide variations to your training data using the variants feature of some of the flavours, but not apply them when validating or testing the network.

For example, to manipulate the ground plane from the Ground orientation flavour only in training:

```yaml
dataset:
  training:
    paths:
      - dataset/training.tfrecord
    config:
      orientation:
        variations:
          height: { mean: 0, stddev: 0.05 }
          rotation: { mean: 0, stddev: 0.0872665 }
```

To make a simple dataset you can follow the instructions in [Quick Start Guide](readme/quickstart.md) or read the code in [training/dataset.py](training/dataset.py)

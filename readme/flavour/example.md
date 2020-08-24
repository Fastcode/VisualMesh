# Example
Example flavours describes how the input image data will be provided to the network.
They provide two essential functions.
Firstly, via their `input` function, they describe how to generate the initial input data loaded from the dataset.
Secondly, they take the projection from the visual mesh that describes which pixels are targeted, and load those specific pixels from the input image.

## Image
Image flavour takes an image that is stored in the dataset and provides it to the visual mesh.
It can accept a wide variety of compressed image formats (anything that is accepted by [tf.image.decode_image](https://www.tensorflow.org/api_docs/python/tf/io/decode_image))

### Dataset Keys
```python
"image": bytes[1] # a compressed image (png, jpg, gif, etc)
```

### Configuration
Any or all of the variations can be left out if you do not wish to apply image variations when running.
All variations are done with pixel values as floats which means that they go from 0->1 not 0->255.

```yaml
example:
  type: Image
  config:
    variations:
      # Adjust the brightness `x + delta`
      brightness:  { mean: 0, stddev: 0.05 }
      # Adjust the contrast `(x - mean) * delta + mean`
      contrast: { mean: 1, stddev: 0.05 }
      # Convert to hsv, adjust the hue by a value from [-1 -> 1] and back to rgb
      hue: { mean: 0, stddev: 0.05 }
      # Convert to hsv, multiply saturation by value and convert back to rgb
      saturation: { mean: 0, stddev: 0.05 }
      # Adjust the gamma `gain * x**gamma`
      gamma:
        gamma: { mean: 1, stddev: 0.05 }
        gain: { mean: 1, stddev: 0.05 }
```

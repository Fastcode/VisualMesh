import tensorflow as tf
import matplotlib.pyplot as plt
import yaml

image_path = "images/image0000002.jpg"
config_path = "augmentations_example.yaml"

# Decode image into tensor
image = open(image_path, 'rb').read()
image_tensor = tf.image.decode_jpeg(image)

# Get augmentations
with open(config_path, "r") as file:
    data = yaml.safe_load(file)
augmentations = data["example"]["config"]["augmentations"]

# Apply the augmentations that were listed
if "brightness" in augmentations:
    v = augmentations["brightness"]
    image_tensor = tf.image.adjust_brightness(image_tensor, tf.random.truncated_normal(shape=(), **v))
if "contrast" in augmentations:
    v = augmentations["contrast"]
    image_tensor = tf.image.adjust_contrast(image_tensor, tf.random.truncated_normal(shape=(), **v))
if "hue" in augmentations:
    v = augmentations["hue"]
    image_tensor = tf.image.adjust_hue(image_tensor, tf.random.truncated_normal(shape=(), **v))
if "saturation" in augmentations:
    v = augmentations["saturation"]
    image_tensor = tf.image.adjust_saturation(image_tensor, tf.random.truncated_normal(shape=(), **v))
if "gamma" in augmentations:
    v_gamma = augmentations["gamma"]["gamma"]
    v_gain = augmentations["gamma"]["gain"]
    image_tensor = tf.image.adjust_gamma(
        image_tensor, tf.random.truncated_normal(shape=(), **v_gamma), tf.random.truncated_normal(shape=(), **v_gain)
    )

# Display image using matplotlib
plt.imshow(image_tensor)
plt.axis('off')
plt.show()

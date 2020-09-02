# Training
Training in the visual mesh code is handled by Tensorflow 2.0.
You perform training by creating a configuration file `config.yaml`, putting it an output directory and running training on that directory.
The training will then train the network and put the output in the same directory as the configuration file.
The output of the network will be a trained weights file and TensorBoard log files.

## Configuration File
This configuration file will describe how the network should be trained and what options should be set.
Most of these options are described as part of the flavour system which you can read about on the [Architecture Page](readme/architecture.md) and [Dataset Page](readme/dataset.md) or in the [Example Configuration File](example_net.yaml).

`Note:` batch size is not set in the training section but is instead handled by the dataset section.

The specific options which are relevant for the training process are documented below.

### Network Section
The design of the network is configured in the `network: { structure: {}}` section of the yaml file.
The network is defined using a graph of elements that you can assemble until a final node `output`.
Currently, for the case of the inference system, they are only able to handle feed-forward networks of Dense and Convolutional layers with no skip connections.
Anywhere in the network, you can use the variable `$output_dims` and this will be replaced with the dimensionality of labels you are using.

There will be two default inputs provided, `X` and `G` that you can use as inputs to your network layers.
These correspond to the input values from the image, and the graph indices that form the graph network.
`GraphConvolution` layers always require `G` as an input along with the output of the previous layer.

For example a very simple two layer network would look like the following
```yaml
network:
  structure:
    g1: { op: GraphConvolution, inputs: [X, G] }
    d1: { op: Dense, inputs: [g1], options: { units: 16, activation: selu, kernel_initializer: lecun_normal } }
    g2: { op: GraphConvolution, inputs: [d1, G] }
    output: { op: Dense, inputs: [g2], options: { units: $output_dims, activation: softmax } }
```

### Training Section
The training section contains the details on how the training will proceed.
This includes the following settings

#### `training: batches_per_epoch`
Batches per epoch can either be set to a number or left out entirely.
If it is set, it is the number of batches that will be processed to be considered an epoch.
This is useful if you have a large dataset and want to have your validation steps update more frequently.
If you don't set this field then the entire dataset will be used as an epoch.

#### `training: epochs`
This is the number of epochs that will be trained for before finishing.
If `batches_per_epoch` is not set it will be the number of times the dataset is seen.
Otherwise, it will be the number of `batchs_per_epoch` loops to do.

#### `training: optimiser`
This selects which optimiser will be used and sets any non-learning rate parameters that it has.
The current options are Adam, SGD and Ranger (RAdam with Lookahead).
Any option that is accepted by the constructor of these is available as configuration options.

[Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)
[SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD)
[RectifiedAdam](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/RectifiedAdam)

#### `training: learning_rate`
The learning rate options is for setting the learning rate scheduler.
It provides several different options for how to control the learning rate during training.

- `static` will keep the same learning rate for the entire training.
With static, you only need to set the learning rate as an option
- `lr_finder` is not designed to be used for training and instead will run through the learning rate finder algorithm to find an upper bound for valid learning rates
- `one_cycle` will use the one cycle learning rate scheduler, which will start the learning rate at some low value, ramp it up to a higher value, and then back down.

## Running training
Running training is done by executing the `./mesh.py train` command on the output folder with `config.yaml` in it.
For example when training using docker:
```sh
./docker run --gpus all -u $(id -u):$(id -g) -it --rm --volume $(pwd):/workspace visualmesh:latest ./mesh.py train <path/to/output>
```

This will run until the target number of epochs has been reached.

### TensorBoard
When running training the network will output the training outputs to TensorBoard in the output directory.
To see the progress and graphs of training you can run `tensorboard --log_dir=<path_to_output>`

Depending on the flavour you use, there will be different graphs and images that are shown in TensorBoard.

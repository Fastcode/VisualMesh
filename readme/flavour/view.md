# View
The view controls how multiple cameras can be used together to make a multi-view visual mesh.
It is able to add a prefix to all the dataset elements used by other flavours and provide a combine function at the end to apply the multi-view transformations.
Often the expanded keys will have some overlap and different views will have the same data.
In this case, you can use the keys field in the dataset to remap the features from multiple views back to a single input.

## Monoscopic
Monoscopic view is the traditional single camera visual mesh mode with no fancy multi camera options.
There is no configuration for this view and it does not impact the keys required by other flavour components.

### Dataset Keys
No dataset keys are required for Monoscopic view
```
None
```

### Configuration
```yaml
view:
  type: Monoscopic
  config: {}
```

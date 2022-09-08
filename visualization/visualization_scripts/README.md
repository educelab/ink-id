# List of Scripts
- gradcam.py

# gradcam.py
This script is an implementation of 3D Grad-CAM loosely based on the 2D version shown in [Ulyanin: Implementing Grad-CAM in PyTorch](https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82)  The original Grad-CAM paper is found [here](https://arxiv.org/abs/1610.02391)

## Requirements
If visualization is desired, Plotly (see above for more information) must be installed.

## Input Data
* CNN architecture: the current implementation assumes encoder (for CNN layers) and decoder (for densely connected layers)
* Pre-trained model: a saved model containing pre-trained weights. Typically a .pt file
* Subvolume data: must be either a 3D numpy array or a directory containing .tif files.

## Output Data
* heatmap (3D numpy array)
* (optional) reverse_heatmap (3D numpy array)
* gradcam.png
* (optional) GradcamReverse.png
* (optional) SimpleSubvolume.png
* (optional) Superimposed.png


## Common Usage
```
g = InkidGradCam(encoder, decoder, path_to_pretrained_model)

# If encoder layer names are unknown
g.print_encoder()

# Register hooks
g.register_hooks(layer_name='layer')

# Push subvolume through
g.push_subvolume_through(input_dir=input_dir_path, [args])

# If quick visualization is desired
g.save_images(output_dir, [args])
```


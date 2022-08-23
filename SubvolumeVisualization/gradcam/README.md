# Introduction

# Contents
* inkid_gradcam.py: Given a CNN model, pre-trained weights, and a subvolume data, produces a 3D rendered image of GradCam.

# Filename Organization (for display filtering)

## For gradcam images
```
{dataset}-{group}-{column}-{truth}-{sample num}-{pretrained model}-{prediction}-{image-type}.png
```
example: `dCarbonPhantom-gInterpolated-c6-tNoink-s3-w13434_36000-p0-iGradcamReverse.png`

(everything up to pretrained-model is given

### Current options for each name-part
```
{
	dataset(d):["CarbonPhantom"],
	group(g): ["Interpolated", "NearestNeighbor"]
	column(c): int
	truth(t): ["Ink", "NoInk","IronGall"],
	sample num(s): int
	image-type(i): ["Plotlymono", "Plotlycolor", "Plotlymono360", "Plotlycolor360",
						   "Ytcolor", "Ytcolor360", "Gradcam", 
						   "GradcamReverse", "Ytlegend",
						   "SimpleSubvolume", "Superimposed"]
	pretrained-model(w): int(id)_int(steps)
	prediction(p): int(0 or 1)
}
```
Metafile for each image should be saved with the same name (the only difference being
the file extension)


# inkid_gradcam.py
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
testcase = InkidGradCam(encoder, decoder, path_to_pretrained_model)

# If encoder layer names are unknown
testcase.print_encoder()

# Register hooks
testcase.register_hooks(layer_name='layer')

# Push subvolume through
testcase.push_subvolume_through(input_dir=input_dir_path, [args])

# If quick visualization is desired
testcase.save_images(output_dir, [args])
```

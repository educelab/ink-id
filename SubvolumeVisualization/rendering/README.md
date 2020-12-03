# Introduction
The directory holds Python scripts for visualizing each subvolume data in a 
human recognizable way. 

# Contents
* volume_renderer.py: produces 3D volume rendered images of subvolume.
* inkid_gradcam.py: Given a CNN model, pre-trained weights, and a subvolume data, produces a 3D rendered image of GradCam.

# Filename Organization (for display filtering)
## For subvolume images 
```
{dataset}-{group}-{column}-{truth}-{sample num}-{image-type}.png
```
example: `dCarbonPhantom-gInterpolated-c6-tNoink-s3-iPlotlymono.png`

(Everything up to sample-num is given)

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
						   "GradcamReverse", "Ytcolorbar",
						   "SimpleSubvolume", "Superimposed"]
	pretrained-model(w): int(id)_int(steps)
	prediction(p): int(0 or 1)
}
```
Metafile for each image should be saved with the same name (the only difference being
the file extension)

# volume_render.py 

## Requirements
The two main visualization libraries used in this script are
* [Plotly](https://plotly.com/python/3d-volume-plots/)
* [The yt project](https://yt-project.org/doc/visualizing/volume_rendering.html)

### Orca
Plotly uses Orca for visualization. Unfortunately getting Orca to work on Jupyter
Notebook takes a few extra steps as documented [here](https://plotly.com/python/orca-management/).

In case that fails to resolve the issue, here is how it was done: 

```
wget https://github.com/plotly/orca/releases/download/v1.3.1/orca-1.3.1.AppImage

mv orca-1.3.1.AppImage /home/mhaya2/anaconda3/envs/vis/bin/

cd /home/mhaya2/anaconda3/envs/vis/bin/

chmod +x orca-1.3.1.AppImage

###  Install following packages
sudo apt-get install desktop-file-utils

sudo apt-get install libgtk2.0-0 
sudo apt-get install libgconf-2-4 
sudo apt-get install xvfb
sudo apt-get install chromium-browser

### Create a file named orca in /home/USERNAME/myvenv/bin/ with the following content:
vim /home/mhaya2/anaconda3/envs/vis/bin/orca

#!/bin/bash

xvfb-run -a orca-1.3.1.AppImage  "$@"

# Then in the code, include the following
import plotly.io as pio
pio.orca.config.executable = '/home/mhaya2/anaconda3/envs/vis/bin/orca'
pio.orca.config.use_xvfb = True
pio.orca.config.save()

pio.write_image(fig, file="test.png")

```

## Input Data
* subvolume data: 

## Output Data
* .png files
* .mp4 files
* .json files (for metadata)

## Common Usage
```
# For Plotly rendition
testcase_pl = volumerenderer.Plotly3D(input_dir, [args])
fig = testcase_pl.setup_graph(colorscale='Rainbow'|'Greys', [args])

## save static image files
testcase_pl.save_image(fig, output_dir, [args])

## save an animated file
testcase_pl.animated_full_rotation(fig, output_dir [args])

# For Yt rendition
testcase_yt = volumerenderer.yt3D(input_dir, [args])
scene = testcase_yt.setup_graph(output_dir, [args])

## save static image files
testcase_yt.save_image(scene, output_dir, [args])

## save an animated file
testcase.animated_full_rotation(scene, n_steps=120)
```

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

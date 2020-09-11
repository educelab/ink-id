# Introduction
The directory holds Python scripts for visualizing each subvolume data in a 
human recognizable way. 

# Contents
* volume_renderer.py: produces 3D volume rendered images of subvolume.
* inkid_gradcam.py: Given a CNN model, pre-trained weights, and a subvolume data, produces a 3D rendered image of GradCam.
* test_volumerenderer.py: a simple test script for testing volume_renderer.py
* test_gradcam.py: a simple test script for testing inkid_gradcam.py
* test_combined.py: a simple test script for using inkid_gradcamp.py to producea heatmap and test_volumerenderer.py to visualize it in various ways.

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

## Common Usage


# inkid_gradcam.py
## Requirements

## Input Data

## Output Data

## Common Usage


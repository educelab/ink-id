# plotly_3d.py
#
# Given a numpy 3D array, generate 3D volume rendering using Ploty 
#


import os
import numpy as np
from pathlib import Path
from PIL import Image
import kaleido
import math
import plotly.graph_objects as go
import io
import argparse
import miscellaneous_functions as misc


def render_plotly(input_arr,
                  name='dataset',
                  direction='z', 
                  opacity=0.3,
                  opacityscale=0.3,
                  surface_count=5,
                  colorscale="jet", 
                  isomax=40000,
                  isomin=0,
                  outfile=None):
    '''
    Renders plotly 3D volume image.
    Args:
        input_arr(numpy array): numpy 3D array
        name(str):
        direction(str): 'x', 'y', or 'z'
        opacity(float):
        opacityscale(float):
        surface_count(int):
        colorscale(str):
        isomax(int):
        isomin(int):
        outfile(str): path/to/output/file.png
    Return:
        io.Bytes
    '''

    size_x, size_y, size_z = np.shape(input_arr)

    X, Y, Z = np.mgrid[0:size_x, 0:size_y, 0:size_z]
    
    vol = go.Volume(
          name=name,
          x = X.flatten(),
          y = Y.flatten(),
          z = Z.flatten(),
          value = input_arr.flatten(),
          opacity = opacity,
          opacityscale = opacityscale,
          surface_count = surface_count,
          colorscale=colorscale,
          isomax = isomax,
          isomin = isomin,
          #slices_z = dict(show=True, locations=[10]),
        )
    fig = go.Figure(data=vol)


    def generate_ticks(axis, interval, size_um=None):
        vals=[]
        ticks =[]
    
        if not size_um:
            size_um=1
        for i in range(0, axis, interval):
            vals.append(i)
            ticks.append(i*size_um)
      
        return (vals, [str(tick) for tick in ticks])
    

    x_vals, x_ticks = generate_ticks(size_x, 8)
    y_vals, y_ticks = generate_ticks(size_y, 8)
    z_vals, z_ticks = generate_ticks(size_z, 8)
   
    if direction == 'x':
        up_direction = dict(x=1, y=0, z=0)
        eye = dict(x=1.25, y=1.25, z=1.25)
    elif direction == 'y':
        up_direction = dict(x=0, y=1, z=0)
        eye = dict(x=1.25, y=1.25, z=1.25)
    else: # directoin == 'z'
        up_direction = dict(x=0, y=0, z=1)
        eye = dict(x=-1.25, y=1.25, z=1.25)

    fig.update_layout(title=name,
                      scene = dict(
                        xaxis = dict(
                            ticktext=x_ticks,
                            tickvals=x_vals),
                        yaxis = dict(
                            ticktext=y_ticks,
                            tickvals=y_vals),
                        zaxis = dict(
                            ticktext=z_ticks,
                            tickvals=z_vals)),
                     scene_aspectmode='data',
                     scene_camera = dict(up=up_direction, eye=eye),
                     )
    
    if outfile:
        fig.write_image(outfile, engine="kaleido")
        
    plotly_bytes = fig.to_image(format="png")
   
    return io.BytesIO(plotly_bytes)


def render_all_axes(input_array, outfile):

    images = []
    
    # Render 3D Images
    for direction in ['x', 'y', 'z']:
        plotly_img = render_plotly(input_array, direction=direction)
        images.append(plotly_img)
 
    
    # Convert the byte arrays to viewable images and concatenate
    graphs = [Image.open(img) for img in images]
    print("image Files length: ", len(graphs))
       
    img_size = [0,0] # (width, height)
    
    for graph in graphs: 
        if graph.width > img_size[0]:
            img_size[0] = graph.width
        img_size[1] = img_size[1] + graph.height
        
    
    summary_img = Image.new('RGB', (img_size[0], img_size[1]))
    current_pos = [0,0]
    
    ##  Stitch all graphs together
    for graph in graphs:
        summary_img.paste(graph,  current_pos)
        # bring the current position down by the pasted image's height
        current_pos = (0, current_pos[1] + graph.height)
    
    
    summary_img.save(outfile)
    
    for img in images:
        img.close()
        
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", help="path to input .npy file")
    parser.add_argument("--output", help="output file path")
    args = parser.parse_args()

    input_array = misc.numpy_binary_to_array(args.input)
    output_file = args.output

    #render_plotly(input_array, outfile=output_file)
    render_all_axes(input_array, output_file)



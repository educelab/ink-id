# vol_render_matplotlib_slices.py 

import os
import numpy as np
from pathlib import Path
from PIL import Image
import re
import matplotlib.pyplot as plt
import math
import io
import argparse
import miscellaneous_functions as misc


# Helper function
def render_along_axis(input_arr, direction, cmap_choice="jet", imgs_in_row=6, 
                            plt_title=None, outfile=None):
    '''
    Render matplotlib heatmap graph slices for one direction

    Args:
        input_arr(numpy array): 3D array
        direction(str): 'x', 'y', or 'z'
        cmap_choice(str):
        imgs_in_row(int):
        plt_title(str):
        outfile(str): path/to/output/img/file.png

    Return:
        io.Bytes(): consisting of a png image
    '''

    size_x, size_y, size_z = np.shape(input_arr)

    if direction == 'x':
        img_count, img_width, img_height = size_x, size_y, size_z
    elif direction == 'y':
        img_count, img_width, img_height = size_y, size_z, size_x
    elif direction == 'z':
        img_count, img_width, img_height = size_z, size_y, size_x

    print("img_count: ", img_count)
    row_count = math.ceil(img_count/imgs_in_row)

    print("row_count: ", row_count)
    figsize = (imgs_in_row*img_width/15, row_count*img_height/12) 
    # divisor is just for a friendly size. May need to be smaller

    # set up the graph
    f, ax_arr = plt.subplots(row_count, imgs_in_row, figsize=figsize)

    for j, row in enumerate(ax_arr):
        for i, ax in enumerate(row):
            if j*imgs_in_row+i < img_count:
                if direction == 'x':
                    ax.imshow(input_arr[j*imgs_in_row+i, :, :],cmap=cmap_choice)
                    ax.set_title(f'x-slice {j*imgs_in_row+i}')
                elif direction == 'y':
                    ax.imshow(input_arr[:,j*imgs_in_row+i, :],cmap=cmap_choice)
                    ax.set_title(f'y-slice {j*imgs_in_row+i}')
                else:  # z
                    ax.imshow(input_arr[:,:, j*imgs_in_row+i], cmap=cmap_choice)
                    ax.set_title(f'z-slice {j*imgs_in_row+i}')

    if plt_title:
        f.suptitle(plt_title, fontsize=12)

    if outfile:
        # Save as an output file only if desired.
        plt.savefig(outfile, format='png')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(f)
    buf.seek(0)

    return buf


def render_along_all_axes(input_arr, cmap_choice="jet", imgs_in_row=6,
                          outfile='output.png'):
    '''
    Render matplotlib heatmap graph slices for all three directions 

    Args:
        input_arr(numpy array): 3D array
        cmap_choice(str):
        imgs_in_row(int):
        outfile(str): path/to/output/img/file.png

    Return:
        None (saves an image file)
    '''

    # Render 2D Images
    images = []
    for direction in ['x', 'y', 'z']:
        slice_img = render_along_axis(
                            input_arr, direction, 
                            cmap_choice=cmap_choice,
                            imgs_in_row=imgs_in_row,
                            plt_title=f'{direction}-axis')
        images.append(slice_img)
    
    
    # Convert the byte arrays to viewable images and concatenate
    graphs = [Image.open(img) for img in images]
    
    # For debugging only
    print("image Files length: ", len(graphs))
    
    
    img_size = [0,0] # (width, height)
    
    for graph in graphs: 
        if graph.width > img_size[0]:
            img_size[0] = graph.width
        img_size[1] = img_size[1] + graph.height
    
    # For debugging only
    print("width: ", img_size[0])
    print("height: ", img_size[1])
    
    slices_img = Image.new('RGB', (img_size[0], img_size[1]))
    current_pos = [0,0]
    
    ##  Stitch all graphs together
    for graph in graphs:
        slices_img.paste(graph,  current_pos)
        # bring the current position down by the pasted image's height
        current_pos = (0, current_pos[1] + graph.height)
    
  
    slices_img.save(outfile)
    
    for img in images:
        img.close()
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", help="path to a .npy file")
    parser.add_argument("--output", help="output file path")
    args = parser.parse_args()

    input_array = misc.numpy_binary_to_array(args.input)
    output_file = args.output

    render_along_all_axes(input_array, outfile=output_file)


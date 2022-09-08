# yt_3d.py
# 3D volume rendering using Yt
# 
import argparse
import numpy as np
import math
import cv2
import json
import yt as yt
from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper
from yt.visualization.volume_rendering.camera import Camera
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import miscellaneous_functions as misc

def render_graph(input_array, outfile="None", 
                field='attenuation', 
                voxel_size_um = 12.0,
                atten_min_intensity=10000, 
                atten_max_intensity=35000, 
                scale=10.0, 
                colormap='gist_rainbow', 
                trans_min_intensity=2.5, 
                trans_max_intensity=17.5, 
                trans_midpoint=10.35):
    '''
    Load data to yt and configure the details
    Args:
        input_array(numpy array): 3D array size (x, y, z)
        outfile(str): file name for render output. If none, no file will be saved
    Output:
        yt.visualization.volume_rendering.scene.Scene
    '''
    info_log = {}

    data = {field: input_array}
    print("data: ", data)

    bbox = np.array([[0.0, d * voxel_size_um] 
                    for d in data[field].shape])

    ds = yt.load_uniform_grid(data, 
            data[field].shape, 
            length_unit='um', bbox=bbox, nprocs=1)

    # Volume render
    sc = yt.create_scene(ds, field=field)
    
    sc.annotate_axes()
    sc.annotate_domain(ds, color=[1, 1, 1, 0.1])
    source = sc.get_source()
    source.set_log(False)
    
    # TF Helper setup
    vol_min = np.min(data[field])
    vol_max = np.max(data[field])
    tfh = source.tfh
    tfh.set_log(False)
    tfh.set_bounds((vol_min, vol_max))

    tfh.build_transfer_function()
    
    vol_high = np.quantile(data[field],0.999)
    vol_low = np.quantile(data[field],0.01)
    
    # Linear TF (SV)
    
    if field == 'attenuation':
        tf_min_intensity = atten_min_intensity
        tf_max_intensity = atten_max_intensity
        tf_min_opacity = vol_min   # Seth's original
        tf_max_opacity = vol_max
        
        def linramp(vals, minval, maxval):
            vs = tf_min_intensity + vals * (tf_max_intensity - tf_min_intensity)
            return np.maximum(np.minimum(
                (vs - tf_min_opacity) / (tf_max_opacity - tf_min_opacity), maxval), minval)
        
        tfh.tf.nbins = 512
        tfh.tf.map_to_colormap(mi=vol_low, ma=vol_high, scale = scale, 
                                colormap=colormap, scale_func=linramp)
        
        
    #elif field == 'transform':

    #    tfh.tf.nbins = 512
    #    tf_min_intensity = trans_min_intensity
    #    tf_max_intensity = trans_max_intensity
    #    midpoint = trans_midpoint
    #    tfh.tf.map_to_colormap(mi=tf_min_intensity, ma=midpoint, colormap='RAINBOW')
    #    tfh.tf.map_to_colormap(mi=midpoint, ma=tf_max_intensity, colormap='RAINBOW_r')
    
   
    if outfile:

        render_scale = 0.5
        sc.camera.resolution = (np.array((1080, 1080)) * render_scale).astype(int)
        
        # Save the colorbar and the rendering
        tfh.plot(fn=f'{outfile}_legend.png')
        sc.save(f'{outfile}_yt.png', sigma_clip=2.5)

        # Log info.
        info_log['field']=field
        info_log['atten_min_intensity']=atten_min_intensity
        info_log['atten_max_intensity']=atten_max_intensity
        info_log['scale']=scale
        info_log['colormap']=colormap
        info_log['trans_min_intensity']=trans_min_intensity
        info_log['trans_max_intensity']=trans_max_intensity
        info_log['trans_midpoint']=trans_midpoint

        with open(f"{outfile}.json", "w") as f: 
            json.dump(info_log, f, indent=2)

    return sc


# TODO: Code below  needs to be edited to make it work.
def animated_full_rotation(scene, output_dir, axes=[1,1,1], n_steps=120, fps=10, 
                           filename="result"):
    '''
    Save an mp4 file showing rotated yt 3D rendering images.
    Input: 
        scene:
        axes(list):
        n_steps(int): how many images there should be for each 360 rotation
        fps(int): 
    Output: None
    '''

    info_log = {}

    # Strip the trailing '/'
    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    render_scale = 0.5
    scene.camera.resolution = (np.array((1080, 1080)) * render_scale).astype(int)

    images = []

    if axes[0]:
        for _ in tqdm(scene.camera.iter_rotate(theta=np.pi * 2, 
                                n_steps=n_steps, rot_vector=(1,0,0))):
            im = scene.render()
            images.append(im)
    
    if axes[1]:
        for _ in tqdm(scene.camera.iter_rotate(theta=np.pi * 2, 
                                n_steps=n_steps, rot_vector=(0,1,0))):
            im = scene.render()
            images.append(im)
    
    if axes[2]:
        for _ in tqdm(scene.camera.iter_rotate(theta=np.pi * 2, 
                                n_steps=n_steps, rot_vector=(0,0,1))):
            im = scene.render()
            images.append(im)

    # Save file
    imagefile = f'{filename}-iYtcolor360'
    
    def sigma_clip(img: np.ndarray, s: float) -> np.ndarray:
        nz = img[:, :, :3][img[:, :, :3].nonzero()]
        max_val = nz.mean() + s * nz.std()
        alpha = img[:, :, 3]
        out = np.clip(img[:, :, :3] / max_val, 0.0, 1.0)
        out = np.concatenate([out, alpha[..., None]], axis=-1)
        return out

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    size = (images[0].shape[1], images[0].shape[0])
    out = cv2.VideoWriter(f'{output_dir}/{imagefile}', fourcc, fps, size)
    out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100.0)
    for im in tqdm(images):
        im = sigma_clip(im, 2.5)
        frame = cv2.cvtColor(np.clip((im * 255), 0, 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
        frame = cv2.flip(frame, 0)
        frame = np.rot90(frame, 3)
        # cv2_imshow(frame)
        out.write(frame)
    out.release()

    info_log['mp4 file']=f'{imagefile}.mp4'
    info_log['n_steps']=n_steps
    info_log['fps']=fps
    
    with open(f"{output_dir}/{imagefile}.json", "w") as outfile: 
        json.dump(info_log, outfile, indent=2)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", help="path to input .npy file")
    parser.add_argument("--output", help="output file path")

    args = parser.parse_args()

    input_array = misc.numpy_binary_to_array(args.input)
    output_file = args.output

    if output_file[-4:] == ".png":
        output_file = output_file[:-4]

    render_graph(input_array, output_file)

import numpy as np
import re
import math
import cv2
import glob
import json
import plotly.io as pio
import plotly.graph_objects as go
import yt as yt
from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper
from yt.visualization.volume_rendering.camera import Camera
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
from PIL import Image
from datetime import datetime
from tqdm import tqdm


class VolumeRenderer:
    '''
    Superclass for the two following inherited classes.
   
    '''

    def __init__(self, output_dir, input_dir=None, subvolume=None):
        '''
        Input:
            output_dir(str): output directory
            input_dir(str): directory path without the trailing '/'
            subvolume(numpy.ndarray):
        '''

        # Strip the trailing '/'
        if output_dir[-1] is '/':
            output_dir = output_dir[:-1]

        # Must have either input_dir or subvolume
        if not input_dir and not subvolume.any():
            print("must provide either input_dir(path for a directory with image \
                  slices) or subvolume (3D numpy array)")
    
        # If input_dir is given, create a subvolume matrix
        if input_dir:
            # Strip the trailing '/'
            if input_dir[-1] is '/':
                input_dir = input_dir[:-1]

            subvolume = self.load_data(input_dir)
            self.input_dir = input_dir  

        else:
            self.input_dir = None
        
        subvolume_fft = np.real(np.fft.fftn(subvolume))   # Fournier Conversion?
        subvolume_fft = np.log(np.abs(np.fft.fftshift(subvolume_fft)))   # Not sure

        self.subvolume = {'attenuation': subvolume, 'transform': subvolume_fft}
        self.voxel_size_um = 12.0
        self.output_dir = output_dir

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.images = []

        # A JSON log will be produced for each output file
        self.log = {}
        self.log['input_data']=input_dir
        self.log['output_dir']=output_dir


    def load_data(self, input_dir):
        '''
        loads *.tif image files into numpy 3D array
        Input:
            input_dir(str): directory contaning *.tif images
        Returns:
            np.ndarray (3D)
        '''
        dataset = Path(input_dir)
        files = list(dataset.glob('*.tif'))
        files.sort(key=lambda f: int(re.sub(r'[^0-9]*', "", str(f))))
    
        subvolume = []
        for f in files:
            i = Image.open(f)
            subvolume.append(np.array(Image.open(f), dtype=np.float32))
  
        return np.array(subvolume)


class Plotly3D(VolumeRenderer):
    def __init__(self, output_dir, input_dir=None, subvolume=None): 
        VolumeRenderer.__init__(self, output_dir, input_dir, subvolume)
        self.log['graph']='plotly'

    def setup_graph(self, field='attenuation', min_val=0, max_val=None, 
                    opacity=0.3, surface_count=20,
                     colorscale='Rainbow', opacityscale=None):
        '''
        Input:
            field(str): 'attenuation' or 'transform'
            min_val(int):
            max_val(int):
            opatity(float): float between 0 and 1. Needs to be small to see through all surfaces
            surface_count(int): needs to be large number for good volume rendering
            colorscale(str): matplotlib or plotly color scale 
            opacityscale(list): e.g.[ [0,0], [max_val, 0], [max_val/4, 1], [max_val, 1]],
        Returns:
            plotly.graph_objs._figure.Figure
    
        '''
        X, Y, Z = np.mgrid[0:self.subvolume[field].shape[0], 
                0:self.subvolume[field].shape[1], 
                0:self.subvolume[field].shape[2]]
   
        maxval = max_val if max_val else np.amax(self.subvolume[field])

        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=self.subvolume[field].flatten(),
            isomin=min_val,
            isomax=maxval, 
            opacity=opacity,
            surface_count=surface_count,
            colorscale= colorscale,
            opacityscale=opacityscale,
        ))

        self.log['field']=field
        self.log['min_val']=min_val
        self.log['max_val']=maxval
        self.log['opacity']=opacity
        self.log['surface_count']=surface_count
        self.log['colorscale']=colorscale
        self.log['opacityscale']=opacityscale
        return fig


    def test_image(self, fig, filename=None,
                   up_cfg=None, center_cfg=None, eye_cfg=None, title=None):
        '''
        Save plotly figure in a given in a given directory
        Input:
            fig(plotly.graph_objs._figure.Figure): 
            filename(str): file name to be given to the image for saving.
            up_cfg(tuple): for camera
            center_cfg(tuple): for camera
            eye_cfg(tuple): for camera
            show(bool): if True, it will display on Jupyter Notebook
            title(str): appears at the top of the image
    
        '''
        # This may be  necessary for orca to work
        #pio.orca.config.executable = '{path to orca--perhaps inside conda env}'
        #pio.orca.config.use_xvfb = True
        #pio.orca.config.save()
    

        # Adjust the camera position and display orientation
        camera = dict(
            up=dict(x=up_cfg[0], y=up_cfg[1], z=up_cfg[2]) if up_cfg else dict(x=0, y=0, z=1),
            center=dict(x=center_cfg[0], y=center_cfg[1], z=center_cfg[2]) if center_cfg 
                        else dict(x=0, y=0, z=0),
            eye=dict(x=eye_cfg[0], y=eye_cfg[1], z=eye_cfg[2]) if eye_cfg 
                    else dict(x=2.5, y=2.5, z=2.5),
        )
    
        fig.update_layout(scene_camera=camera, scene_dragmode='orbit', 
                          title=title, font=dict(size=9))
    
        # TODO: add more figure update options
    
        # Save file
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f'{timestamp}.png'
        pio.write_image(fig, f'{self.output_dir}/{filename}')
    
        self.log['output file']=filename
        self.log['up_cfg']=up_cfg if up_cfg else (0,0,1)
        self.log['center_cfg']=center_cfg if center_cfg else (0,0,0)
        self.log['eye_cfg']=eye_cfg if eye_cfg else (2.5, 2.5, 2.5)
        
        with open(f"{self.output_dir}/{timestamp}.json", "w") as outfile: 
            json.dump(self.log, outfile, indent=2)

    def animated_full_rotation(self, fig, 
                            rotate_angle=10, transition_angle=10, camera_distance=2.5, 
                            title=None, fps=30, filename=None):
        '''
        Saves animated version of 360 rotation along x, y, z axes. 
        Input:
            fig(plotly.graph_objs._figure.Figure):
            rotate_angle(int): interval of rotated angles along the axes
            transition_angle(int): interval of rotated angles during transitions
            camera_distance(int): camera distance (the greater the further)
            title(str): title that appears at the top of the image
            filename(str): name of the mp4 output file 
        Returns: None
        '''

        title = title if title else self.input_dir

        # x-axis
        for angle in range(0, 361, rotate_angle):
            adj_angle = angle-90
            
            camera = dict(
                up=dict(x=1, y=0, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0,
                        y=math.sin(math.radians(adj_angle))*camera_distance,
                        z=math.cos(math.radians(adj_angle))*camera_distance)
                )
        
            fig.update_layout(scene_camera=camera, scene_dragmode='orbit', 
                                title=title, font=dict(size=9))

            img_bytes = fig.to_image(format='png')
            self.images.append(img_bytes)
                
        # Transition between x and y (a)
        for angle in range(transition_angle, 90, transition_angle):  # 0 and 90 would be redundant
            camera = dict(
                up=dict(x=1, y=0, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=math.sin(math.radians(angle))*camera_distance,
                        y=math.cos(math.radians(angle))*camera_distance*(-1),
                        z=0)
                )
        
            fig.update_layout(scene_camera=camera, scene_dragmode='orbit', 
                                title=title, font=dict(size=9))

            img_bytes = fig.to_image(format='png')
            self.images.append(img_bytes)
        
        
        # y-axis
        for angle in range(0, 361, rotate_angle):
            camera = dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=math.cos(math.radians(angle))*camera_distance,
                        y=0,
                        z=math.sin(math.radians(angle))*camera_distance)
                )
        
            fig.update_layout(scene_camera=camera, scene_dragmode='orbit', 
                                title=title, font=dict(size=9))

            img_bytes = fig.to_image(format='png')
            self.images.append(img_bytes)

        
        # Transition between y and z (b)
        for angle in range(transition_angle, 90, transition_angle):  # 0 and 90 would be redundant
            camera = dict(
                up=dict(x=0,
                        y=math.cos(math.radians(angle))*camera_distance,
                        z=math.sin(math.radians(angle))*camera_distance),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=camera_distance, y=0, z=0))
        
            fig.update_layout(scene_camera=camera, scene_dragmode='orbit', 
                                title=title, font=dict(size=9))

            img_bytes = fig.to_image(format='png')
            self.images.append(img_bytes)
            
        
        # z-axis
        for angle in range(0, 361, rotate_angle):
            adj_angle = 360-angle

            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=math.cos(math.radians(adj_angle))*camera_distance,
                        y=math.sin(math.radians(adj_angle))*camera_distance,
                        z=0)
                )
        
            fig.update_layout(scene_camera=camera, scene_dragmode='orbit', 
                                title=title, font=dict(size=9))

            img_bytes = fig.to_image(format='png')
            self.images.append(img_bytes)
    
        img_array = []
        
        # Create an animated mp4 file and save
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f'{timestamp}.mp4'

        for image in self.images:
            nparr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{self.output_dir}/{filename}', fourcc, fps, size)
         
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        
        self.log['mp4 file']=filename
        self.log['rotate_angle'] = rotate_angle
        self.log['transition_angle']=transition_angle
        self.log['camera distance'] = camera_distance
        self.log['fps']=fps
    
        with open(f"{self.output_dir}/{timestamp}.json", "w") as outfile: 
            json.dump(self.log, outfile, indent=2)
        

class yt3D(VolumeRenderer):
    def __init__(self, output_dir, input_dir=None, subvolume=None): 

        VolumeRenderer.__init__(self, output_dir, input_dir, subvolume)
        self.log['graph']='yt'


    def setup_graph(self, field='attenuation', atten_min_intensity=10000, 
                    atten_max_intensity=35000, scale=10.0, colormap='gist_rainbow', 
                    trans_min_intensity=2.5, trans_max_intensity=17.5, 
                    trans_midpoint=10.35):
        '''
        Load data to yt and configure the details
        Input: TODO
        Output:
            yt.visualization.volume_rendering.scene.Scene
        '''
        bbox = np.array([[0.0, d * self.voxel_size_um] 
                        for d in self.subvolume[field].shape])

        ds = yt.load_uniform_grid(self.subvolume, 
                self.subvolume[field].shape, 
                length_unit='um', bbox=bbox, nprocs=1)

        # Volume render
        sc = yt.create_scene(ds, field=field)
        
        sc.annotate_axes()
        sc.annotate_domain(ds, color=[1, 1, 1, 0.1])
        source = sc.get_source()
        source.set_log(False)
        
        # TF Helper setup
        vol_min = np.min(self.subvolume[field])
        vol_max = np.max(self.subvolume[field])
        tfh = source.tfh
        tfh.set_log(False)
        tfh.set_bounds((vol_min, vol_max))

        tfh.build_transfer_function()
        
        vol_high = np.quantile(self.subvolume[field],0.999)
        vol_low = np.quantile(self.subvolume[field],0.01)
        
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
            
            
        elif field == 'transform':

            tfh.tf.nbins = 512
            tf_min_intensity = trans_min_intensity
            tf_max_intensity = trans_max_intensity
            midpoint = trans_midpoint
            tfh.tf.map_to_colormap(mi=tf_min_intensity, ma=midpoint, colormap='RAINBOW')
            tfh.tf.map_to_colormap(mi=midpoint, ma=tf_max_intensity, colormap='RAINBOW_r')
        
        
        #Save the colorbar image 
        tfh.plot(fn=f'{self.output_dir}/yt_colorbar.png')

        self.log['field']=field
        self.log['atten_min_intensity']=atten_min_intensity
        self.log['atten_max_intensity']=atten_max_intensity
        self.log['scale']=scale
        self.log['colormap']=colormap
        self.log['trans_min_intensity']=trans_min_intensity
        self.log['trans_max_intensity']=trans_max_intensity
        self.log['trans_midpoint']=trans_midpoint

        return sc

    def save_image(self, scene, filename=None):
        '''
        Save a yt test image, 
        Input:
            scene(yt.visualization.volume_rendering.scene.Scene):
        '''
        render_scale = 0.5
        scene.camera.resolution = (np.array((1080, 1080)) * render_scale).astype(int)
        
        # Save file
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f'{timestamp}.png'

        scene.save(f'{self.output_dir}/{filename}', sigma_clip=2.5)

        with open(f"{self.output_dir}/{timestamp}.json", "w") as outfile: 
            json.dump(self.log, outfile, indent=2)

    def animated_full_rotation(self, scene, axes=[1,1,1], n_steps=360, fps=30, 
                               filename=None):
        '''
        Save an mp4 file showing rotated yt 3D rendering images.
        Input: 
            scene:
            axes(list):
            n_steps(int): how many images there should be for each 360 rotation
            fps(int): 
        Output: None
        '''

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
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f'{timestamp}.mp4'
        
        def sigma_clip(img: np.ndarray, s: float) -> np.ndarray:
            nz = img[:, :, :3][img[:, :, :3].nonzero()]
            max_val = nz.mean() + s * nz.std()
            alpha = img[:, :, 3]
            out = np.clip(img[:, :, :3] / max_val, 0.0, 1.0)
            out = np.concatenate([out, alpha[..., None]], axis=-1)
            return out

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        size = (images[0].shape[1], images[0].shape[0])
        out = cv2.VideoWriter(f'{self.output_dir}/{filename}', fourcc, fps, size)
        out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100.0)
        for im in tqdm(images):
            im = sigma_clip(im, 2.5)
            frame = cv2.cvtColor(np.clip((im * 255), 0, 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
            frame = cv2.flip(frame, 0)
            frame = np.rot90(frame, 3)
            # cv2_imshow(frame)
            out.write(frame)
        out.release()

        self.log['mp4 file']=filename
        self.log['n_steps']=n_steps
        self.log['fps']=fps
        
        with open(f"{self.output_dir}/{timestamp}.json", "w") as outfile: 
            json.dump(self.log, outfile, indent=2)


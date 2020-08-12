import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union
import numpy as np
from pathlib import Path
from PIL import Image
import re
import os
import plotly.graph_objects as go

class HookedCNN:
    '''
    Creates a 3D CNN with a pre-trained model  and hooks to be used by GradCam
    '''
    
    def __init__(self, encoder, decoder, trained_model):
        '''
        trained_model(str): 'trained_model/x.pt' ...
        '''

        # Create a model
        self.encoder = encoder
        self.decoder = decoder
        self.model=torch.nn.Sequential(encoder, decoder)
    
        # placeholder for the gradients
        self.gradients = None
    
        # placeholder for the activations
        self.activations = None
    
        # Register hooks  TODO: Make it more generic
        self.model[0].conv4.register_forward_hook(self.save_activations)
        self.model[0].conv4.register_backward_hook(self.save_gradients)
    
        self.trained_model = trained_model

    def save_activations(self, module, input, output):    
        '''
        Inputs:
            input(tuple): 
            output(tensor):
        '''
        #print('input size:', input[0].size(), '    <---- activations')
        self.activations = input[0]

    def save_gradients(self, module, grad_input, grad_output):
        '''
        TODO:
        '''

        #print('grad_input size:', grad_input[0].size(), '    <---- gradients' )
        self.gradients = grad_input[0]


    def print_model(self):
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
  
    def load_model(self):
        self.model.load_state_dict(torch.load(self.trained_model, 
                                map_location=torch.device('cpu')), strict=False)
        self.model.eval()


class InkidGradCam:
    '''
    3D GradCam implementation specifically for Inkid subvolumes.
    
    Attributes
    ----------
    input_dir: str
        Name of the input directory containing *.tif files
    output_dir: str
        Name of the output directory where generated images will be saved.
    encoder:
    decoder:
    saved_model:
    model:
    subvolume: numpy.ndarray
        3D array representing an Inkid subvolume.
    prediction: int
        0 or 1
    heatmap: torch tensor
        3D tensor representing "heat" distribution

    Methods
    -------
    load_data(input_dir):
        Loads data from an input directory containing *.tif files.
        To be used when input_dir, rather than subvolume, is given.
    load_model(encoder, decoder, saved_model):
        Loads the saved model into the CNN architecture.
    print_model()
    push_subvolume_through()
    calculate_heatmap():
    visualize_heatmap():
        Generates a Plotly rendition of the 3D heatmap.
    animate_heatmap():
        Generates a 360 rotated view of the heatmap.
    visalize_subvolume():
        Generates a Plotly rendition of the input subvolume.
    superimposed_heatmap():
        Generates a superimposed image of subvolume and heatmap.
        Needs to be edited later.
    get_prediction(): 
    get_gradients():
    get_activations():
    get_heatmap():
    get_model():
        Getter functions for corresponding attributes.
    '''

    def __init__(self, output_dir, encoder, decoder, saved_model, input_dir=None, 
                 subvolume=None):
        '''
        Input:
            output_dir(str):
            encoder:
            decoder:
            saved_model(str):
            input_dir(str): directory path where input *.tif files are found.
            subvolume(numpy.ndarray): Alternatively, numpy 3D array representing
                                      the subvolume.
        '''

        # Strip the trailing '/'
        if output_dir[-1] is '/':
            output_dir = output_dir[:-1]

        self.output_dir = output_dir

        # Must have either input_dir or subvolume
        if not input_dir and not subvolume:
            print("must provide either input_dir(path for a directory with image \
                  slices) or subvolume (3D numpy array)")

        # If input_dir is given, create a subvolume matrix.
        if input_dir:
            # Strip the trailing '/'
            if input_dir[-1] is '/':
                self.input_dir = input_dir[:-1]

            self.subvolume = torch.from_numpy(self.load_data())

        else:
            self.input_dir = None
            self.subvolume = torch.from_numpy(subvolume)

        self.encoder = encoder
        self.decoder = decoder
        self.saved_model = saved_model

        # Placeholder for model, prediction, heatmap
        self.model = None
        self.prediction = None
        self.heatmap = None

        def load_model(self):
            self.model = HookedCNN(self.encoder, self.decoder, self.saved_model)
            self.model.load_model()

        def print_nodel(self):
            self.model.print_model()

        def push_subvolume_through(self):
            self.prediction = self.model(self.subvolume).argmax(dim=1).item()

            self.model(self.subvolume)[:, self.prediction, :, :]backward()
            
            self.activations = self.model.activations
            self.gradients = self.model.gradients


        def calculate_heatmap(self):
            pooled_gradients = torch.mean(self.gradients, dim[0,2,3,4])

            # weight the channels by corresponding gradients
            for i in ragne(pooled_gradients.size()): # should be 8
                self.activations[:, i, :, :, :] *= pooled_gradients[i]

            # average the channels of the activations
            heatmap = torch.mean(activations, dim=1).squeeze()

            # relu on top of the heatmap (we are not insterested in negative values)
            heatmap = np.maximum(heatmap.detach(), 0)
            
            # normalize the heatmap
            self.heatmap /= torch.max(heatmap)

        def visualize_heatmap(self):
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)

            X, Y, Z = np.mgrid[0:12:, 0:12, 0:12]
            values = self.heatmap
            
            gradient_map = go.Figure(data=go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=values.flatten(),
                isomin=0.1,
                isomax=1.0,
                opacity=0.2, # needs to be small to see through all surfaces
                surface_count=8, # needs to be a large number for good volume rendering
                colorscale = 'rainbow'
                ))
            
            gradient_map.update_layout(showlegend=False)
            
            gradient_map.write_image(f"{self.output_dir}/gradient_map.png")
           

        def animate_heatmap(self):
            # TODO

        def visalize_subvolume(self):
            '''
            Generates a Plotly rendition of the input subvolume.
            '''
            subvol = torch.squeeze(self.subvolume)
            
            X, Y, Z = np.mgrid[0:48, 0:48, 0:48]


            subvolume_map = go.Figure(data=go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=subvol.flatten(),
                isomin=0.1,
                isomax=torch.max(subvol).item()*0.9,
                opacity=0.3, # needs to be small to see through all surfaces
                surface_count=8, # needs to be a large number for good volume rendering
                colorscale = 'Greys'
                ))
            
            subvolume_map.update_layout(showlegend=False)
            subvolume_map.write_image(f"{selfoutput_dir}/subvolume_map.png")

        def superimposed_heatmap(self):
            gradient_img = cv2.imread(f'{self.output_dir}/gradient_map.png')
            subvolume_img = cv2.imread(f'{self.output_dir}/subvolume_map.png')
            
            superimposed_img = cv2.addWeighted(gradient_img, 0.35, subvolume_img, 0.65, 0)
            cv2.imwrite(f'{self.output_dir}/superimposed.png', superimposed_img)            
            
        
        def load_data(self):
            '''
            A helper function that loads data from an input directory containing 
            *.tif files.
            Returns:
                numpy.ndarray
            '''
    
            dataset = Path(self.input_dir)
            files = list(dataset.glob('*.tif'))
            files.sort(key=lambda f: int(re.sub(r'[^0-9]*', "", str(f))))
        
            subvolume = []
            for f in files:
                i = Image.open(f)
                subvolume.append(np.array(Image.open(f), dtype=np.float32))
        
            return np.array(subvolume)

        def get_prediction():
            # TODO
            pass

        def get_gradients():
            # TODO
            pass

        def get_activations():
            # TODO
            pass
    
        def get_heatmap():
            # TODO
            pass

        def get_model():
            # TODO


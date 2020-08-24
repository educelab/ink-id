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


class InkidGradCam:
    '''
    3D GradCam implementation specifically for Inkid subvolumes.
    
    Attributes
    ----------
    input_dir(str):  Name of the input directory containing *.tif files
    output_dir(str): Name of the output directory where generated images will be saved.
    encoder:
    decoder:
    saved_model:
    subvolume (numpy.ndarray): 3D array representing an Inkid subvolume.
    prediction(int): 0 or 1
    heatmap(torch tensor): 3D tensor representing "heat" distribution

    Methods
    -------
    TODO:
    '''

    def __init__(self, output_dir, encoder, decoder, saved_model, input_dir=None, 
                 subvolume=None):
        '''
        Input:
            output_dir(str):
            encoder:
            decoder:
            saved_model(str): path to the .pt file
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
            self.input_dir = input_dir[:-1] if input_dir[-1] is '/' else input_dir

            self.subvolume = torch.from_numpy(self.load_data())

        else:
            self.input_dir = None
            self.subvolume = torch.from_numpy(subvolume)

        # add two more axes
        self.subvolume = self.subvolume[np.newaxis, np.newaxis, ...]

        self.encoder = encoder
        self.decoder = decoder
        self.saved_model = saved_model
        
        # Place holder for the 3DCNN model with hooks
        self.__net = None
        self.__activations = None
        self.__gradients = None

        # Placeholder for prediction, heatmap
        self.prediction = None
        self.heatmap = None


    def print_encoder(self):
        for name, module in self.encoder.named_children():
            print(name, ": ", module)

    def register_hooks(self, layer_name='conv4'):
        # layer is the name of the child module inside the encoder
        # retried from print_encoder)
        

        def save_activations(module, input, output):
            #print('input size:', input[0].size(), '    <---- activations')
            self.__activations = input[0]


        def save_gradients(module, grad_input, grad_output):
            #print('grad_input size:', grad_input[0].size(), '    <---- gradients' )
            self.__gradients = grad_input[0]


        self.__net = torch.nn.Sequential(self.encoder, self.decoder)

        for name, module in self.__net[0].named_children():
            if name is layer_name:
                module.register_forward_hook(save_activations)
                module.register_backward_hook(save_gradients)
                break

        #self.__net[0].layer.register_forward_hook(save_activations)
        #self.__net[0].layer.register_backward_hook(save_gradients)



    def load_model(self):
        self.__net.load_state_dict(torch.load(self.saved_model, 
                        map_location=torch.device('cpu')), strict=False)
        self.__net.eval()


    def print_nodel(self):
        for param_tensor in self.__net.state_dict():
            print(param_tensor, "\t", self.__net.state_dict()[param_tensor].size())
 

    def push_subvolume_through(self):
        self.prediction = self.__net(self.subvolume).argmax(dim=1).item()

        self.__net(self.subvolume)[:, self.prediction, :, :].backward()
        

        #### This should be unnecessary
        #self.activations = self.__net.activations
        #self.gradients = self.__net.gradients


    def calculate_heatmap(self):
        pooled_gradients = torch.mean(self.__gradients, dim=[0,2,3,4])

        # weight the channels by corresponding gradients
        for i in range(pooled_gradients.size()[0]): # should be 8
            self.__activations[:, i, :, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(self.__activations, dim=1).squeeze()

        # relu on top of the heatmap (we are not insterested in negative values)
        heatmap = np.maximum(heatmap.detach(), 0)
        
        # normalize the heatmap
        self.heatmap = heatmap/torch.max(heatmap)

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
        pass


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
        pass


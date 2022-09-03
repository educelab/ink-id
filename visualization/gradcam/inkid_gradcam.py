import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union
import numpy as np
from pathlib import Path
from PIL import Image
import json
import re
import os
import plotly.graph_objects as go
import cv2
from datetime import datetime

class InkidGradCam:
    '''
    3D GradCam. Current implementation for inkid subvolumes. 
    
    Attributes
    ----------
    encoder(torch.nn.Module): 
        CNN layers
    decoder(torch.nn.Module): 
        Fully-connected layers
    pretrained_model(str): 
        Pre-trained model with weights. Typically a .pt file
    subvolume(torch.Tensor): 
        5D tensor created when a subvolume is given
    output_dir(str): 
        Name of the output directory where generated images will be saved.
    input_dir(str): 
        Name of the input directory if given
    __net(torch.nn.Sequential): 
        Container for encoder and decoder modules
    __activations(torch.Tensor): 
        5D tensor of activations obtained through forward hook
    __gradients(torch.Tensor): 
        5D tensor of activations obtained through backward hook
    prediction(int): 
        0 (no ink) or 1 (ink)
    heatmap(torch.Tensor): 
        3D tensor representing "heat" distribution
    log(dict): 
        Log for metadata

    Methods
    -------
    print_encoder():
        Prints CNN layers in the encoder
    register_hooks(layer_name):
        Creates foward and backward hooks on the given layer_name
    print_final_model():
        Prints the model with hooks
    push_subvolume_through(output_dir, input_dir, subvolume): 
        Pushes subvolume through the model with hooks
    save_images(heatmap, subvolume, superimposed): 
        Saves png images
    animate_heatmap(): 
        TODO?
    __save_metadata(output_dir): 
        Write JSON log and saves it in the output_dir
    __load_data(): 
        Loads subvolume data from a directory containing *.tif files
    get_prediction():
    get_gradients():
    get_activations():

    '''

    def __init__(self, encoder, decoder, pretrained_model, comments=None):
        '''
        Input:
            encoder:
            decoder:
            pretrained_model(str): path to the .pt file
            comments(str): comments that will be added to the metadata.json file
        '''


        # CNN attributes
        self.encoder = encoder
        self.decoder = decoder
        self.pretrained_model = pretrained_model
        
        # Placeholders for subvolume and input
        self.subvolume = None
        self.input_dir = None

        # Placeholders for the 3DCNN model with hooks
        self.__net = None
        self.__activations = None
        self.__gradients = None

        # Placeholders for prediction, heatmap, reverse_heatmap
        self.prediction = None
        self.heatmap = None
        self.reverse_heatmap = None

        # A JSON log will be produced for each output file
        self.log = {}
        self.log['encoder'] = str(self.encoder.__class__.__name__)
        self.log['decoder'] = str(self.decoder.__class__.__name__)
        self.log['pretrained_model'] = self.pretrained_model
        
        if comments:
            self.log['comments'] = comments


    def print_encoder(self):
        '''
        For retrieving names of CNN layers (for registering hooks)
        '''
        for name, module in self.encoder.named_children():
            print(name, ": ", module)


    def register_hooks(self, layer_name='conv4'):
        '''
        Registers forward and backward hooks on a given layer.
        Input:
        layer_name(str): name of the child module inside the encoder
                         (can be obtained through self.print_encoder() method)
        '''

        def save_activations(module, input, output):
            '''
            Method to be used for the forward hook
            '''
            #print('input size:', input[0].size(), '    <---- activations')
            self.__activations = input[0]


        def save_gradients(module, grad_input, grad_output):
            '''
            Method to be used for the backward hook.
            '''
            #print('grad_input size:', grad_input[0].size(), '    <---- gradients' )
            self.__gradients = grad_input[0]


        self.__net = torch.nn.Sequential(self.encoder, self.decoder)

        for name, module in self.__net[0].named_children():
            if name is layer_name:
                module.register_forward_hook(save_activations)
                module.register_backward_hook(save_gradients)
                break


        # Load the given pre-trained model (.pt file) to set the weights.
        self.__net.load_state_dict(torch.load(self.pretrained_model, 
                        map_location=torch.device('cpu'))['model_state_dict'], strict=False)
        self.__net.eval()

        self.log['hook_layer'] = layer_name 


    def print_final_model(self):
        '''
        Prints the whole architecture.  Must be called after self.register_hooks
        '''
        for param_tensor in self.__net.state_dict():
            print(param_tensor, "\t", self.__net.state_dict()[param_tensor].size())
 


    def push_subvolume_through(self, reverse=False, input_dir=None, subvolume=None):
        '''
        Pushes the subvolume through the model 
        Inputs:
            input_dir(str): directory path where input *.tif files are found.
            subvolume(numpy.ndarray): Alternatively, numpy 3D array representing
                                      the subvolume.
        '''
        # Record the time
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # First, process the subvolume data 
        ## Must have either input_dir or subvolume
        if not input_dir and not subvolume:
            print("must provide either input_dir(path for a directory with image \
                  slices) or subvolume (3D numpy array)")

        ## If input_dir is given, create a subvolume matrix.
        if input_dir:
            # Strip the trailing '/'
            self.input_dir = input_dir[:-1] if input_dir[-1] is '/' else input_dir

            self.subvolume = torch.from_numpy(self.__load_data())
            self.log['input_dir'] = self.input_dir 

        else:
            self.input_dir = None
            self.subvolume = torch.from_numpy(subvolume)
            self.log['input_dir'] = "No input_dir given"

        ## add two more axes
        self.subvolume = self.subvolume[np.newaxis, np.newaxis, ...]


        # Second, push the subvolume through
        output = self.__net(self.subvolume)
        print("Output: ", output)
        self.prediction = output.argmax(dim=1).item()
        self.log['prediction'] = self.prediction
        print("Prediction is: ", self.prediction)


        # Record the non-predicted side for possible later use
        __rejection = abs(self.prediction-1)


        # Backpropagate!
        self.__net(self.subvolume)[:, self.prediction, :, :].backward()


        ## Expression 1 in the paper: Average the channels of the activations
        pooled_gradients = torch.mean(self.__gradients, dim=[0,2,3,4])
        
        
        ## Espression 2 in the paper: Relu on top of the heatmap 
        ### weight the channels by corresponding gradients
        for i in range(pooled_gradients.size()[0]): 
            self.__activations[:, i, :, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.__activations, dim=1).squeeze()

        ### Discard negative values we are not interested in and convert to 
        ### a numpy array.
        heatmap = np.maximum(heatmap.detach(), 0)
        
        ### normalize the heatmap
        self.heatmap = heatmap/torch.max(heatmap)

        

        if reverse:

            ### Calculate the rejection heatmap in a similar manner
            self.__net(self.subvolume)[:, __rejection, :, :].backward()


            ## Expression 1 in the paper: 
            pooled_gradients = torch.mean(self.__gradients, dim=[0,2,3,4])
            
            
            ## Espression 2 in the paper: 
            for i in range(pooled_gradients.size()[0]): 
                self.__activations[:, i, :, :, :] *= pooled_gradients[i]

            __reverse_heatmap = torch.mean(self.__activations, dim=1).squeeze()

            ### Discard negative values 
            __reverse_heatmap = np.maximum(__reverse_heatmap.detach(), 0)
            
            ### normalize the heatmap
            self.reverse_heatmap = __reverse_heatmap/torch.max(__reverse_heatmap)

            
            return (self.heatmap, self.reverse_heatmap)


        else:  # if no reverse map is desired
            return (self.heatmap, None)



    def save_images(self, output_dir, heatmap=True, reverse_heatmap=True, 
                    subvolume=False, superimposed=False, filename="result"):
        '''
        Saves 3 types of images in the output directory.
        Inputs:
            heatmap(bool): image will be saved when set to True.
            reverse_heatmap(bool): (same as above)
            subvolume(bool):  (same as above)
            superimposed(bool):  (same as above)
            filename(str): filename stem to which "{pretrained-model}-{prediction}
                           -{imagetype}.png" will be appended.
        '''

        # This may be  necessary for orca to work
        #pio.orca.config.executable = '{path to orca--perhaps inside conda env}'
        #pio.orca.config.use_xvfb = True
        #pio.orca.config.save()
        
        # Strip the trailing '/' from the output_dir path
        if output_dir[-1] is '/':
            output_dir = output_dir[:-1]

        self.log['output_dir'] = output_dir

        # If output_dir does not exist, create one
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if heatmap:
            cube_size = self.heatmap.size()[0]
            X, Y, Z = np.mgrid[0:cube_size:, 0:cube_size, 0:cube_size]

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
            
            # Create an imagefile with an appropriate name
            imagefile = f'{filename}-p{self.prediction}-iGradcamDefault'
            gradient_map.write_image(f"{output_dir}/{imagefile}.png")
        
            # Save the metadata
            self.__save_metadata(output_dir, filename=imagefile)
            
        if reverse_heatmap:
            cube_size = self.reverse_heatmap.size()[0]
            X, Y, Z = np.mgrid[0:cube_size:, 0:cube_size, 0:cube_size]

            reverse_values = self.reverse_heatmap
            
            gradient_map = go.Figure(data=go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=reverse_values.flatten(),
                isomin=0.1,
                isomax=1.0,
                opacity=0.2, # needs to be small to see through all surfaces
                surface_count=8, # needs to be a large number for good volume rendering
                colorscale = 'rainbow'
                ))
            
            gradient_map.update_layout(showlegend=False)

            # Create an imagefile with an appropriate name
            imagefile = f'{filename}-p{self.prediction}-iGradcamReverse'
            gradient_map.write_image(f"{output_dir}/{imagefile}.png")
        
            # Save the metadata
            self.__save_metadata(output_dir, filename=imagefile)


        if subvolume or superimposed:
            subvol = torch.squeeze(self.subvolume)
            cube_size = subvol.size()[0]        
            X, Y, Z = np.mgrid[0:cube_size, 0:cube_size, 0:cube_size]


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

            # Create an imagefile with an appropriate name
            imagefile = f'{filename}-p{self.prediction}-iSimpleSubvolume'
            gradient_map.write_image(f"{output_dir}/{imagefile}.png")
        
            # Save the metadata
            self.__save_metadata(output_dir, filename=imagefile)

            if superimposed:
                gradient_img = cv2.imread(f'{output_dir}/{filename}-p{self.prediction}-iGradCamDefault.png')
                subvolume_img = cv2.imread(f'{output_dir}/{filename}-p{self.prediction}-iSimpleSubvolume.png')
                
                superimposed_img = cv2.addWeighted(gradient_img, 0.35, subvolume_img, 0.65, 0)

                # Create an imagefile with an appropriate name
                imagefile = f'{filename}-p{self.predictino}-iSuperimposed'
                cv2.imwrite(f'{output_dir}/{imagefile}.png', superimposed_img)            

                # Save the metadata
                self.__save_metadata(output_dir, filename=imagefile)

    def animate_heatmap(self):
        # TODO?
        pass


    def __save_metadata(self, output_dir, filename="gradcam-result"):
        '''
        Helper function for saving metadata (called within save_images())
        '''

        with open(f"{output_dir}/{filename}.json", "w") as outfile: 
            json.dump(self.log, outfile, indent=2)
    

    def __load_data(self):
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


    def get_prediction(self):
        # 0: no ink
        # 1: ink
        return self.prediction


    def get_gradients(self):
        return self.__gradients


    def get_activations(self):
        return self.__activations        




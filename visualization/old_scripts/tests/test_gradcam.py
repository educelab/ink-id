# For testing the inkid_gradcam.py script
import inkid_gradcam as gradcam
import copy_model as model
from datetime import datetime
import numpy as np


now = datetime.now()
input_dir = "<path>"
output_dir = "<path>"+ now.strftime("%Y%m%d%H%M")

encoder = model.EncoderModuleWithAllTheParameters
decoder = model.DecoderModuleWithAllTheParameters
pretrained_model = "<path to a .pt file>"

print("Test constructor")
inkid = gradcam.InkidGradCam(encoder, decoder, pretrained_model)

print("Test print_encoder()")
inkid.print_encoder()

print("Test reigster_hooks()")
inkid.register_hooks(layer_name='<layer name>')

print("Test print_final_model()")
inkid.print_final_model()

print("Test push_subvolume_through")
heatmap = inkid.push_subvolume_through(output_dir, input_dir)

print("Test save_images()")
inkid.save_images()



print("Test get_predition")
print(inkid.get_prediction())

print("Test get_gradients")
print(inkid.get_gradients().size())

print("Test get_activations")
print(inkid.get_activations().size())



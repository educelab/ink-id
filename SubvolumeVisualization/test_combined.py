# For testing the inkid_gradcam.py and volume_renderer.py scripts
# Script uses a file called sample_model.py that contains encoder and decoder


import inkid_gradcam as gradcam
import sample_model as model
import volume_renderer as renderer
from datetime import datetime

now = datetime.now()
input_dir = "<input directory where *.tif files are found. Unncessary if a subvol is passed>"
output_dir = "<output directory>"+ now.strftime("%Y%m%d%H%M")

encoder = model.EncoderWithAllTheParameters
decoder = model.DecoderWithAllTheParameters

pre_trained_model = "<typically a .pt file>"

print("Test constructor")
inkid = gradcam.InkidGradCam(encoder, decoder, pre_trained_model)

print("Test reigster_hooks()")
inkid.register_hooks(layer_name='conv4')

print("Test push_subvolume_through")
heatmap = inkid.push_subvolume_through(output_dir, input_dir)


def RunPlotly(output_dir, input_dir=None, subvol=None):

    graph = renderer.Plotly3D(output_dir, input_dir=input_dir, subvolume=subvol)
    fig = graph.setup_graph()
    graph.test_image(fig)
    #graph.animated_full_rotation(fig,rotate_angle=5, transition_angle=10)

def RunYt(output_dir, input_dir=None, subvol=None):

    graph2 = renderer.yt3D(output_dir, input_dir=input_dir, subvolume=subvol)
    scene = graph2.setup_graph()
    graph2.save_image(scene) 
    #graph2.animated_full_rotation(scene, n_steps=120)


RunPlotly(output_dir, subvol=heatmap.numpy())
RunYt(output_dir, subvol=heatmap.numpy())


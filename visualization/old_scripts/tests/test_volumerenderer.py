# For testing volume_renderer.py script

import volume_renderer as renderer
from datetime import datetime

now = datetime.now()
input_dir = "<path>" 
output_dir = "<path>"+ now.strftime("%Y%m%d%H%M")



def RunPlotly(output_dir, input_dir):

    graph = renderer.Plotly3D(output_dir, input_dir=input_dir)
    fig = graph.setup_graph()
    graph.test_image(fig)
    #graph.animated_full_rotation(fig,rotate_angle=5, transition_angle=10)

def RunYt(output_dir, input_dir):

    graph2 = renderer.yt3D(output_dir, input_dir=input_dir)
    scene = graph2.setup_graph()
    graph2.save_image(scene) 
    #graph2.animated_full_rotation(scene, n_steps=120)


RunPlotly(output_dir, input_dir)
RunYt(output_dir, input_dir)



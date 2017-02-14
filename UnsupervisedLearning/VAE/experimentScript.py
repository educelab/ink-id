'''
experimentScript.py
    - invoke main.py with different hyperparameters
    - also create the directory to save the data
'''

import os
import pdb

saveVideosPath = "/home/volcart/VAE_Layers/experiment/"

for dimension in range(15, 80, 5):
    for stepSize in range(int(dimension/2), dimension+1, int(dimension/2)):
        experimentPath = saveVideosPath + "Dimension-" + str(dimension) + "-Step-" + str(stepSize) + "/"
        os.mkdir(experimentPath)
        experimentVidoePath = experimentPath + "videos/"
        os.mkdir(experimentVidoePath)
        os.system("python main.py " + str(dimension) + " " + str(stepSize) + " " + experimentPath + " " + experimentVidoePath)

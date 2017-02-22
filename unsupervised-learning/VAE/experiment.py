'''
experimentScript.py
    - invoke main.py with different hyperparameters
    - also create the directory to save the data
'''

__author__ = "Kendall Weihe"
__email__ = "kendall.weihe@uky.edu"


import os

saveVideosPath = "/home/volcart/UnsupervisedResults/VAE/"
for dimension in range(25, 80, 5):
    stepSize = int(dimension / 2)
    if stepSize % 2 != 0: stepSize += 1
    experimentPath = saveVideosPath + "Dimension-" + str(dimension) + "-Step-" + str(stepSize) + "/"
    os.mkdir(experimentPath)
    experimentVidoePath = experimentPath + "videos/"
    os.mkdir(experimentVidoePath)
    os.system("python3 main.py " + str(dimension) + " " + str(stepSize) + " " + experimentPath + " " + experimentVidoePath)

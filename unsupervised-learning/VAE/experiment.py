'''
experimentScript.py
    - invoke main.py with different hyperparameters
    - also create the directory to save the data
'''

__author__ = "Kendall Weihe"
__email__ = "kendall.weihe@uky.edu"


import os

savePath = "/home/volcart/UnsupervisedResults/HercFragment/VAE/"
dataPath = "/home/volcart/prelim-InkDetection/HercFragment/resliced/"
for dimension in range(25, 80, 5):
    saveSamplesPath = savePath + "Dimension-" + str(dimension) + "/"
    os.mkdir(saveSamplesPath)
    os.system("python3 main.py " + dataPath + " " + str(dimension) + " " + saveSamplesPath)

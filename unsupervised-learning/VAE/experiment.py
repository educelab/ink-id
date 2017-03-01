'''
experimentScript.py
    - invoke main.py with different hyperparameters
    - also create the directory to save the data
'''

__author__ = "Kendall Weihe"
__email__ = "kendall.weihe@uky.edu"


import os

savePath = "/home/volcart/UnsupervisedResults/CarbonPhantom-Feb2017/Column-6/"
dataPath = "/home/volcart/volumes/packages/CarbonPhantom-Feb2017.volpkg/paths/20170221130948/layered/column-6/"
numCubes = [1000, 500, 250]
dimensions = [25, 50, 75]
for i in range(3):
    saveSamplesPath = savePath + "Dimension-" + str(dimensions[i]) + "/"
    os.mkdir(saveSamplesPath)
    os.system("python3 main.py " + dataPath + " " + str(dimensions[i]) + " " + saveSamplesPath + " " + str(numCubes[i]))

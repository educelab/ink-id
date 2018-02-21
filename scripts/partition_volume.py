import numpy as np
from PIL import Image
import cv2
import os
import re
import pdb

dataPath = "/home/volcart/volumes/packages/CarbonPhantom_MP_2017.volpkg/paths/all-cols/layers_130/"
dataFiles = os.listdir(dataPath)
dataFiles.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

volume = []
for f in dataFiles:
    sliceData = np.array(Image.open(dataPath+f))
    volume.append(sliceData)
volume = np.transpose(volume, (2,1,0))

xCoordinates = [468,831]
yCoordinates = [36,1422]
zCoordinates = [0,volume.shape[2]]

newVolume = volume[xCoordinates[0]:xCoordinates[1], yCoordinates[0]:yCoordinates[1], zCoordinates[0]:zCoordinates[1]]
newVolume = 255 * (newVolume - np.min(newVolume)) / (np.amax(newVolume) - np.min(newVolume))

slicesPath = "/home/volcart/volumes/packages/CarbonPhantom_MP_2017.volpkg/paths/all-cols/cropped/layers_130/"
for i in range(newVolume.shape[2]):
    # predictionSlice = 255 * (newVolume[:,:,i] - np.min(newVolume)) / (np.amax(newVolume) - np.min(newVolume))
    sliceNumber = str(i).zfill(4)
    cv2.imwrite(slicesPath+"/"+str(sliceNumber)+".png", newVolume[:,:,i])

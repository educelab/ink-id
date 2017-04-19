import os
from PIL import Image
import cv2
import numpy as np
import pdb

gtPath = "/home/volcart/volumes/packages/CarbonPhantom_MP_2017.volpkg/paths/all-cols/"
gtFiles = ["GroundTruth-IronGallInk.png", "GroundTruth-CarbonInk.png", "GroundTruth-Papyrus.png"]

xCoordinates = [1611,1896]
yCoordinates = [81,1512]

saveCroppedPath = "/home/volcart/volumes/packages/CarbonPhantom_MP_2017.volpkg/paths/all-cols/col4-ground-truth/"
for gt in gtFiles:
    gtData = np.array(Image.open(gtPath+gt))
    gtData = np.transpose(gtData, (1,0))
    gtData = 255 * (gtData - np.min(gtData)) / (np.amax(gtData) - np.min(gtData))
    croppedGT = gtData[xCoordinates[0]:xCoordinates[1], yCoordinates[0]:yCoordinates[1]]
    cv2.imwrite(saveCroppedPath+gt, croppedGT)

slicePath = "/home/volcart/volumes/packages/CarbonPhantom_MP_2017.volpkg/paths/all-cols/layers_130/37.png"
sliceData = np.array(Image.open(slicePath))
sliceData = np.transpose(sliceData, (1,0))
croppedSlice = sliceData[xCoordinates[0]:xCoordinates[1], yCoordinates[0]:yCoordinates[1]]
croppedSlice = 255 * (croppedSlice - np.min(croppedSlice)) / (np.amax(croppedSlice) - np.min(croppedSlice))
cv2.imwrite(saveCroppedPath+"slice.png", croppedSlice)

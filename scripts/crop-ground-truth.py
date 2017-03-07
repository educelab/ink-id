import os
from PIL import Image
import cv2
import numpy as np
import pdb

gtPath = "/home/volcart/volumes/packages/CarbonPhantom-Feb2017.volpkg/paths/20170221130948/layered/registered/ground-truth/"
gtFiles = ["GroundTruth-IronGallInk.png", "GroundTruth-CarbonInk.png", "GroundTruth-Papyrus.png"]

xCoordinates = [106,570]
yCoordinates = [80,1774]

saveCroppedPath = "/home/volcart/volumes/packages/CarbonPhantom-Feb2017.volpkg/paths/20170221130948/layered/registered/ground-truth/column-6/"
for gt in gtFiles:
    gtData = np.array(Image.open(gtPath+gt))
    gtData = np.transpose(gtData, (1,0))
    gtData = 255 * (gtData - np.min(gtData)) / (np.amax(gtData) - np.min(gtData))
    croppedGT = gtData[xCoordinates[0]:xCoordinates[1], yCoordinates[0]:yCoordinates[1]]
    cv2.imwrite(saveCroppedPath+gt, croppedGT)

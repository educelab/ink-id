import numpy as np
import cv2
import pdb

gtPath = "/home/volcart/volumes/packages/CarbonPhantom-Feb2017.volpkg/paths/20170221130948/layered/registered/ground-truth/column-6/"

gtInk = cv2.imread(gtPath+"GroundTruth-CarbonInk.png")[:,:,0]
gtPapyrus = cv2.imread(gtPath+"GroundTruth-Papyrus.png")[:,:,0]

inkIndices = np.transpose(np.where(gtInk == 255))
papyrusIndices = np.transpose(np.where(gtPapyrus == 0))
# nullIndices = np.transpose(np.where(gtPapyrus == 255))

outputGroundTruth = np.zeros((gtInk.shape[0], gtInk.shape[1]), dtype=np.uint8)

# outputGroundTruth[inkIndices] = 255
# outputGroundTruth[papyrusIndices] = 128
# outputGroundTruth[nullIndices] = 0

for i in range(papyrusIndices.shape[0]):
    outputGroundTruth[papyrusIndices[i,0], papyrusIndices[i,1]] = 0

for i in range(inkIndices.shape[0]):
    outputGroundTruth[inkIndices[i,0], inkIndices[i,1]] = 255

# for i in range(nullIndices.shape[0]):
#     outputGroundTruth[nullIndices[i,0], nullIndices[i,1]] = 0

cv2.imwrite("/home/volcart/Desktop/gt.png", outputGroundTruth)

'''
data.py:
    - input training data
    - save captured data from network
'''

__author__ = "Kendall Weihe"
__email__ = "kendall.weihe@uky.edu"

import numpy as np
import pdb
import os
from PIL import Image
import math
import re

def readData(args):
    files = os.listdir(args["dataPath"])
    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    volume = []
    for f in files:
        sliceData = np.array(Image.open(args["dataPath"]+f))
        volume.append(sliceData)
    volume = np.transpose(volume, (2,1,0))

    # uncomment below line to normalize input (0,1)
    # volume = (volume - np.min(volume)) / (np.amax(volume) - np.min(volume))

    nCubesX = int(math.floor(int(volume.shape[0] - args["x_Dimension"] + args["subVolumeStepSize"]) / args["subVolumeStepSize"]))
    nCubesY = int(math.floor(int(volume.shape[1] - args["y_Dimension"] + args["subVolumeStepSize"]) / args["subVolumeStepSize"]))
    nCubesZ = int(math.floor(int(volume.shape[2] - args["z_Dimension"] + args["subVolumeStepSize"]) / args["subVolumeStepSize"]))
    nCubesTotal = nCubesX * nCubesY * nCubesZ
    inputSubVolumes = np.empty((nCubesTotal, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]))
    overlappingCoordinates = np.empty((nCubesTotal, 3))
    cubeCount = 0
    for i in range(0, volume.shape[0]-args["x_Dimension"]+1, args["subVolumeStepSize"]):
        for j in range(0, volume.shape[1]-args["y_Dimension"]+1, args["subVolumeStepSize"]):
            for k in range(0, volume.shape[2]-args["z_Dimension"]+1, args["subVolumeStepSize"]):
                inputSubVolumes[cubeCount,:,:,:] = volume[i:i+args["x_Dimension"], j:j+args["y_Dimension"], k:k+args["z_Dimension"]]
                overlappingCoordinates[cubeCount,0] = i
                overlappingCoordinates[cubeCount,1] = j
                overlappingCoordinates[cubeCount,2] = k
                cubeCount += 1

    return overlappingCoordinates, inputSubVolumes

def saveSamples(args, overlappingCoordinates, sampleList):
    sampleRates = [1, 2, 4, 4, 2, 1]
    for i in range(len(sampleList)): # iterate through layers
        print("Saving layer number: " + str(i+1))
        for j in range(sampleList[i].shape[4]): # iterate through samples
            print("Saving sample number: " + str(j+1))
            sample = np.zeros((2000,2000,200), dtype=np.float32) # NOTE: calculating individual sample sizes has not been developed yet, so I declare a large empty volume and clip later
            counter = np.zeros((2000,2000,200), dtype=np.float32)
            for k in range(sampleList[i].shape[0]): # iterate through sub volumes
                xCoordinate = int(overlappingCoordinates[k,0] / sampleRates[i])
                yCoordinate = int(overlappingCoordinates[k,1] / sampleRates[i])
                zCoordinate = int(overlappingCoordinates[k,2] / sampleRates[i])
                sample[xCoordinate:xCoordinate+sampleList[i].shape[1], yCoordinate:yCoordinate+sampleList[i].shape[2], \
                        zCoordinate:zCoordinate+sampleList[i].shape[3]] += sampleList[i][k,:,:,:,j]
                counter[xCoordinate:xCoordinate+sampleList[i].shape[1], yCoordinate:yCoordinate+sampleList[i].shape[2], \
                        zCoordinate:zCoordinate+sampleList[i].shape[3]] += 1

            # find the true size of the sample
            for k in range(2000):
                if sample[k,0,0] == 0.0:
                    clipXCoordinate = k
                    break
            for k in range(2000):
                if sample[0,k,0] == 0.0:
                    clipYCoordinate = k
                    break
            for k in range(200):
                if sample[0,0,k] == 0.0:
                    clipZCoordinate = k
                    break

            # clip the sample
            sample = sample[0:clipXCoordinate, 0:clipYCoordinate, 0:clipZCoordinate]
            counter = counter[0:clipXCoordinate, 0:clipYCoordinate, 0:clipZCoordinate]
            sample /= counter # average each voxel

            # save 2D slices
            path = args["saveSamplePath"] + str(i+1) + "/"
            try:
                os.mkdir(path)
            except:
                pass
            path = path + "sample" + str(j+1) + "/"
            os.mkdir(path)
            for k in range(sample.shape[2]):
                sliceNumber = str(k).zfill(4)
                f = args["saveSamplePath"] + str(i+1) + "/sample" + str(j+1) + "/" + sliceNumber + ".jpg"
                if (np.amax(sample) - np.min(sample)) != 0:
                    sliceIm = 255 * ((sample[:,:,k] - np.min(sample)) / (np.amax(sample) - np.min(sample)))
                else:
                    sliceIm = sample[:,:,k]
                sliceIm = np.transpose(sliceIm, (1,0))
                im = Image.fromarray(sliceIm)
                im.convert('RGB').save(f)

            # save video of sample
            videoFile = args["saveVideoPath"] + "layer-" + str(i+1) + "-sample-" + str(j+1) + ".avi"
            sliceFiles = args["saveSamplePath"] + str(i+1) + "/sample" + str(j+1) + "/%04d.jpg"
            os.system("ffmpeg -y -framerate 10 -start_number 0 -i " + sliceFiles + " -vcodec mpeg4 " + videoFile)

    # save one video that spans all neurons
    videoFiles = os.listdir(args["saveVideoPath"])
    videoFiles.sort()
    del videoFiles[0] # delete the video allSamples.avi that already exists TODO: search if file exists and then delete
    ffmpegConcatFileData = []
    for f in videoFiles:
        ffmpegConcatFileData.append("file " + args["saveVideoPath"] + f)

    # write the list of video files to a config file -- needed for ffmpeg
    ffmpegFileList = open(args["ffmpegFileListPath"], 'w')
    for line in ffmpegConcatFileData:
        ffmpegFileList.write(line+"\n")
    ffmpegFileList.close()

    # generate the video
    outputConcatVideo = args["saveVideoPath"] + "allSamples.avi"
    os.system("ffmpeg -y -f concat -safe 0 -i " + args["ffmpegFileListPath"] + " -c copy " + outputConcatVideo)

    # TODO:
        # evntually: test out adding text to videos with neuron info?

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
    # volume = (volume - np.min(volume)) / (np.amax(volume) - np.min(volume))

    x = int(math.floor(int(volume[:,0,0].shape[0])/args["x_Dimension"]))
    y = int(math.floor(int(volume[0,:,0].shape[0])/args["y_Dimension"]))
    z = int(math.floor(int(volume[0,0,:].shape[0])/args["z_Dimension"]))
    n_cubes =  z * x * y
    inputSubVolumes = np.empty((n_cubes, args["x_Dimension"], args["y_Dimension"], args["z_Dimension"]))
    cubeCount = 0
    for i in range(0, volume.shape[0]-args["x_Dimension"], args["x_Dimension"]):
        for j in range(0, volume.shape[1]-args["y_Dimension"], args["y_Dimension"]):
            for k in range(0, volume.shape[2]-args["z_Dimension"], args["z_Dimension"]):
                inputSubVolumes[cubeCount,:,:,:] = volume[i:i+args["x_Dimension"], j:j+args["y_Dimension"], k:k+args["z_Dimension"]]
                cubeCount += 1

    return x, y, z, inputSubVolumes

def saveSamples(args, sampleList):
    for i in range(sampleList.shape[0]): # iterate through layers
        for j in range(sampleList[i].shape[3]): # iterate through samples
            for k in range(sampleList[i].shape[2]): # iterate through slices
                sliceNumber = str(k).zfill(4)
                f = args["saveSamplePath"] + str(i+1) + "/sample" + str(j+1) + "/" + sliceNumber + ".jpg"
                if (np.amax(sampleList[i]) - np.min(sampleList[i])) != 0:
                    sliceIm = 255 * ((sampleList[i][:,:,k,j] - np.min(sampleList[i])) / (np.amax(sampleList[i]) - np.min(sampleList[i])))
                else:
                    sliceIm = sampleList[i][:,:,k,j]
                sliceIm = np.transpose(sliceIm, (1,0))
                im = Image.fromarray(sliceIm)
                im.convert('RGB').save(f)

            # save video of each sample
            videoFile = args["saveVideoPath"] + "layer-" + str(i+1) + "-sample-" + str(j+1) + ".avi"
            sliceFiles = args["saveSamplePath"] + str(i+1) + "/sample" + str(j+1) + "/%04d.jpg"
            os.system("ffmpeg -y -framerate 10 -start_number 0 -i " + sliceFiles + " -vcodec mpeg4 " + videoFile)

    # save one video that spans all neurons
    videoFiles = os.listdir(args["saveVideoPath"])
    videoFiles.sort()
    del videoFiles[0]
    ffmpegConcatFileData = []
    for f in videoFiles:
        ffmpegConcatFileData.append("file " + args["saveVideoPath"] + f)

    # write the list of video files to a config file -- for ffmpeg
    ffmpegFileList = open(args["ffmpegFileListPath"], 'w')
    for line in ffmpegConcatFileData:
        ffmpegFileList.write(line+"\n")
    ffmpegFileList.close()

    outputConcatVideo = args["saveVideoPath"] + "allSamples.avi"
    os.system("ffmpeg -y -f concat -safe 0 -i " + args["ffmpegFileListPath"] + " -c copy " + outputConcatVideo)

    # TODO:
        # evntually: test out adding text to videos with neuron info?

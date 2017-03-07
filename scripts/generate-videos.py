import os
import pdb

rootPath = "/home/volcart/UnsupervisedResults/CarbonPhantom-Feb2017/Column-6/Dimension-25-numNuerons-32-norm-zNotDownsampled/"

layers = os.listdir(rootPath)
layers = [ x for x in layers if x.isdigit() ]

saveVideosPath = rootPath + "videos/"
for l in layers:
    neurons = os.listdir(rootPath+l)
    for n in neurons:
        slices = rootPath+l+"/"+n+"/%04d.jpg"
        os.system("ffmpeg -y -framerate 10 -start_number 0 -i " + slices + " -vcodec mpeg4 " + saveVideosPath+"layer-"+l+"-neuron-"+n+".mp4")


ffmpegFileListPath = rootPath + "videoList.txt"
videoFiles = os.listdir(saveVideosPath)
videoFiles.sort()

ffmpegConcatFileData = []
for f in videoFiles:
    ffmpegConcatFileData.append("file " + saveVideosPath + f)
# write the list of video files to a config file -- needed for ffmpeg
ffmpegFileList = open(ffmpegFileListPath, 'w')
for line in ffmpegConcatFileData:
    ffmpegFileList.write(line+"\n")
ffmpegFileList.close()
# generate the video
outputConcatVideo = saveVideosPath + "allSamples.avi"
os.system("ffmpeg -y -f concat -safe 0 -i " + ffmpegFileListPath + " -c copy " + outputConcatVideo)

# H1 Variational Autoencoder (VAE)
## Based on paper from [here](https://arxiv.org/pdf/1312.6114.pdf)
### Author: Kendall Weihe

##### General abstraction
A autoencoder (neural network) consists of an encoder that tries to encode data in a unique way such that the decoder can generate the original data.
Through this process, individual neurons of the network learn features of the dataset. The purposes of this autoencoder is to see if neurons are able to learn
to identify ink features that are not visible to the human eye. This network will train on any volume (specified by a path) and capture the output of individual neurons
over the entire volume.

##### Capturing neuron outputs happens in three steps
* first, the program saves neuron samples to a unique directory
    * the naming scheme of the directory must be `1/sample1/*.jpg` where the first `1/` means the first layer in the autoencoder and the `sample1/` means the first sample of that layer
* second, the program uses `ffmpeg` to generate a video for each sample
* third, the program constructs on large video of all the video samples
    * this is to make the analyzing process quicker -- each video is created at 10 fps

##### Configuring the network

Adjust the following hash table to configure the network.

```[python]
args = {
    "dataPath": "/home/volcart/Pictures/LampBlackTest-2016.volpkg/paths/20161205161113/flattened_i1/",
    "x_Dimension": 50,
    "y_Dimension": 50,
    "z_Dimension": 50,
    "subVolumeStepSize": 25,
    "learningRate": 0.001,
    "batchSize": 10,
    "dropout": 0.75,
    "trainingIterations": 10001,
    "analyzeStep": 5000,
    "displayStep": 5,
    "saveSamplePath": "/home/volcart/VAE_Layers/",
    "saveVideoPath": "/home/volcart/VAE_Layers/videos/",
    "ffmpegFileListPath": "/home/volcart/VAE_Layers/concatFileList.txt"
}
```

###### Breaking it down
* `dataPath` is the path to the volume -- in 2D slices
* `x_Dimension` and the others are the dimensions of the subvolume that will be sent through the network
* `subVolumeStepSize` is how much each input subvolume will overlap -- i.e. the above configuration, each subvolume will be overlapped by (most likely) 5 other subvolumes
* `analyzeStep` is the epoch step at which neuron output will be captured
* `displayStep` is the epoch step at which the loss and accuracy will be printed to the terminal
* `saveSamplePath` is the path where the samples will be saved -- outlined in **Capturing neuron outputs happens in three steps**
* `saveVideoPath` is the path where individual sample videos will be saved
* `ffmpegFileListPath` is the path to the configuration file needed for concatenating videos with `ffmpeg` -- outlined [here](https://trac.ffmpeg.org/wiki/Concatenate)


**Loss functions**
Notice that the decoder has two loss functions, where one is always commented out. There is the cross entropy function and the root mean squared error (RMSE) function.
Uncomment one and comment the other to test different loss functions -- I recommend cross entropy.

**Number of neurons**
This VAE is a convolutional neural network (CNN) with 6 convolutional layers. To adjust the number of neurons, adjust the `filter` variables in the following hash table (located in `model.py`).

```[python]
networkParams = {
    "nFilter0": 1,
    "nFilter1": 2,
    "nFilter2": 4,
    "nFilter3": 8,
    "zSize": 20
}
```

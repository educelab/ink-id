# Configuration:

This application uses a dictionary called `config` as a sort-of global configuration variable throughout the entirety of the app. Defining `config` makes experimentation simpler and more accessible. For example, write a `experiment.py` script and pass configuration variables to the `main.py` program.

**Example `config` dictionary:**:
```[python]
config = {
    # FLAGS
    "surface_segmentation": False,
    "mulitpower": True,
    "crop": True,
    "addRandom": True,
    "useJitter": False,

    # PATHS
    "trainingDataPath": "/home/volcart/volumes/test/input_MP/",
    "groundTruthFile": "/home/volcart/volumes/test/gt.png",
    "surfaceDataFile": "/home/volcart/volumes/test/surface.png",
    "savePredictionPath": "/home/volcart/volumes/test/MP-single-channel_model/",
    "saveModelPath": "/home/volcart/volumes/test/MP-single-channel_model/",

    # DATA
    "cropX_low": 550,
    "cropX_high": 600,
    "cropY_low": 450,
    "cropY_high": 500,
    "numVolumes": 2,
    "x_Dimension": 10,
    "y_Dimension": 10,
    "z_Dimension": 30,
    "stride": 1,
    "scalingFactor": 1.0,
    "randomStep": 10,
    "randomRange" : 200,
    "addAugmentation": True,
    "surfaceCushion" : 5,
    "jitterRange" : [-3, 3],

    # MODEL
    "numChannels": 1,
    "n_Classes": 2,
    "learningRate": 0.0001,

    # SESSION
    "epochs": 2,
    "batchSize": 24, # NOTE: for multipower single channel, this must be a multiple of numVolumes
    "predictBatchSize": 24, # NOTE: for multipower single channel, this must be a multiple of numVolumes
    "dropout": 0.75,
    "predictStep": 200000,
    "displayStep": 100,
    "saveModelStep": 100,
    "graphStep": 1000,
}
```

**TODO**: change the configuration dictionary to a configuration file

# Assumptions:

We have developed `data.py` and `ops.py` for multipurpose use. It is up to *you* to write a `main.py` file and even add additional models to `model.py`

# data.py:

`__init__()`:
  - reads volume(s) -- more than one if `config["multipower"] == True`
  - reads ground truth
    - single PNG registered image
  - reads surface segmentation image -- if `config["surface_segmentation"] == True`
  - crops volume(s), ground truth, and surface segmentation -- if `config["crop"] == True`

`getTrainingCoordinates()`:
  - returns coordinates of training sub-volumes
  - only returns coordinates if the sub-volume contains at least 90% of either ink or papyrus
  - shuffles the coordinates
  - should be called before each epoch

`getRandomTestCoordinates()`:
  - same logic as `getTrainingCoordinates()`, except returns coordinates within the testing bounds

`get2DPredictionCoordinates()`:
  - returns all [x,y] coordinates of the volume given the stride

`get3DPredictionCoordinates()`:
  - returns all [x,y,z] coordinates of the volume given the stride

`getSamples()`:
  - accepts coordinates (in either [x,y] or [x,y,z]), returns samples at those coordinates
  - accepts surface segmentation coordinates
  - adds data augmentation if `config["addAugmentation"] == True`
  - adds random noise if `config["addNoise"] == True`
  - pads sub-volumes with zeros if around edges
  - scales sub-volumes based on `config["scalingFactor"]`
  - returns tuple of shape `[batch_size, x, y, z, num_volumes]`

`totalPredictions()`:
  - returns the total number of sub-volumes

`initPredictionImages()`:
  - to be called before making predictions for 2D images

`initPredictionVolumes()`:
  - to be called before making predictions for 3D volumes

`reconstruct2D()`:
  - to be called after making a single prediction batch for `initPredictionImages()`

`reconstruct3D()`:
  - to be called after making a single prediction batch for `initPredictionVolumes()`

`savePredictionImages()`:
  - saves prediction images for `initPredictionImages()`

`savePredictionVolumes()`:
  - saves predictions for `initPredictionVolumes()`

# ops.py:

`graph()`:
  - graphs:
    - test accuracy
    - test loss
    - train accuracy
    - train loss
    - test precision
    - train precision
  - saves to image, location defined in `config`

`getRandomBrick()`:
  - returns a sub-volume filled with random noise
  - distribution of values are aligned with the median value of the actual sub-volume

`edge()`:
  - accepts coordinate, sub-volume size, and axis shape
  - returns True if coordinate is located on volume edge
  - else returns False

`findEdgeSubVolume()`:
  - returns a edge sub-volume with zero padding

`bounds()`:
  - returns the bounds tuples for X and Y dimensions
    - used in finding training vs testing coordinates

`findRandomCoordinates()`: (deprecated -- not used)
  - returns x, y, z coordinates in random locations within the bounds parameters

`splice()`:
  - splices edges off sub-volume that has been scaled beyond the size of the defined sub-volume size

`customReshape()`: (deprecated -- not used)
  - reshapes a list of multipower volumes to shape [batch, x, y, z, num_volumes]

`getSpecString()`:
  - returns string with date, time, and sub-volume size

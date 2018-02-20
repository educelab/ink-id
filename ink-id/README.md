# Configuration:
Running this code requires three main configuration categories: input data, network structure, and parameters

### Input data
This is probably the greatest barrier to getting a network up and running.
The input data required is:
* a directory containing volumetric slices. NOTE: this directory (3DCNN) assumes top-down slices, ex. dimensions (1440 x 285)
* an image of surface locations to use (ex. 2530 x 1440 x 1)
* a ground truth label (ex. 2530 x 1440 x 1)

### Network structure
This is controlled in `model.py` and should be fairly simple to "plug and play" with different structures.
For example, these lines add a fifth convolutional layer with 32 filters to the existing four-layer network
```[python]
conv4 = slim.batch_norm(slim.convolution(conv3, args["neurons"][3], args["receptiveField"], stride=[2,2,2]))
conv5 = slim.batch_norm(slim.convolution(conv4, 32, args["receptiveField"], stride=[2,2,2]))

net = tf.nn.dropout(slim.fully_connected(slim.flatten(conv5), args["n_Classes"], activation_fn=None), args["dropout"])
```

### Parameters
All other parameters are set in the `args` dictionary, and should be self-explanatory.
Some are also hardcoded in `main.py`, 
**Example `args` dictionary:**:
```[python]
args = {
    ### Input configuration ###
    "trainingDataPath" : "/home/jack/devel/volcart/small-fragment-data/flatfielded-slices/",
    "surfaceDataFile": "/home/jack/devel/volcart/small-fragment-data/polyfit-slices-degree32-cush16-thresh20500/surface.tif",
    "groundTruthFile": "/home/jack/devel/volcart/small-fragment-data/ink-only-mask.tif",
    "savePredictionPath": "/home/jack/devel/volcart/predictions/3dcnn/",
    "x_Dimension": int(sys.argv[1]),
    "y_Dimension": int(sys.argv[1]),
    "z_Dimension": int(sys.argv[2]),
    "surfaceCushion" : 20,

    ### Network configuration ###
    "receptiveField" : [3,3,3],
    "learningRate": 0.0001,
    "batchSize": 30,
    "predictBatchSize": 200,
    "dropout": 0.5,
    "neurons": [4, 8, 16, 32],
    "trainingIterations": 10001,
    "trainingEpochs": 1,
    "n_Classes": 2,

    ### Data configuration ###
    "numCubes" : 500,
    "addRandom" : True,
    "randomStep" : 10, # one in every randomStep non-ink samples will be a random brick
    "randomRange" : 200,
    "useJitter" : True,
    "jitterRange" : [-6, 6],
    "addAugmentation" : True,
    "train_portion" : .6,
    "balance_samples" : True,
    "train_quadrants" : -1, # parameters: 0=test top left (else train) || 1=test top right || 2=test bottom left || 3=test bottom right
    "trainBounds" : 3, # bounds parameters: 0=TOP || 1=RIGHT || 2=BOTTOM || 3=LEFT
    "grabNewSamples": 20,
    "surfaceThresh": 20400,
    "restrictSurface": False,

    ### Output configuration ###
    "predictStep": 20000,
    "displayStep": 50,
    "overlapStep": 4,
    "predictDepth" : 1,
    "savePredictionFolder" : "/home/jack/devel/volcart/predictions/3dcnn/{}x{}x{}-{}-{}-{}h/".format(
            sys.argv[1], sys.argv[1], sys.argv[2],  #x, y, z
            datetime.datetime.today().timetuple()[1], # month
            datetime.datetime.today().timetuple()[2], # day
            datetime.datetime.today().timetuple()[3]), # hour
    "notes": "Neuron experiment"
}
```

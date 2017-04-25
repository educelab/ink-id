# Configuration:

... how to configure

... i.e. you can run experiments with multiple different configurations by changing the config dictionary

**TODO**: change the configuration dictionary to a configuration file

# Assumptions:

We have developed `data.py` and `ops.py` for mulitpurposes uses. It is up to *you* to write a `main.py` file and even add additional models to `model.py`

# data.py:

...

# ops.py:

...

# Multipower-single-channel:

... how to configure

# Multipower-multi-channel:

... how to configure

# Multipower-multi-network:

... how to configure


# INFO on code... to be filled in later

## @Jack NOTES for data.py:
- `init()`:
  - reads in list of volumes & the ground truth
    - if there is a single volume, then you must specify config["multipower"]: False
    - self.volume is of shape [num_volumes, x, y, z]
    - self.groundTruth is of shape [x, y]
  - if config["crop"] == True: then crops based on config[] crop values
  - if config["surface_segmentation"] == True: then open surface segmentation file


- `getSamples()`:
  - grabs samples of size sub-volume from the defined coordinate positions
    - parameters:
      - config dictionary
      - coordinates
      - FLAGS: addNoise, augment, addJitter, includeEdgeSubVolumes
    - returns batch of shape [batch_size, x, y, z, num_volumes]

*TODO:*
- @Jack create your own `"main"` file and name it appropriately
  - I have already created two of my own --> for testing
- add the following IF conditions based on flag parameters to `getSamples()`
  - addNoise
  - augment
  - addJitter
  - includeEdgeSubVolumes
- in `getTrainingCoordinates()` make sure that we check that there is at least 90% ink or no ink
- do the same thing for getRandomTestCoordinates() ^^^
- add surface option to `zCoordinate` in `getSamples()`
- 3D predictions:
  - NOTE: just add a z-coordinate to the functions `get***Coordinates()`
    - because then the coordinates will be passed to `getSamples()`
    - NOTE: assign `coordinates[i][2]` to `zCoordinate` in `getSamples()`
  - for 3D recontructions:
    - pass num_images = z_depth as a parameters to `initPredictionImages()`
    - you may need to make minor adjustment to `main` file
- add metric variables such as `self.all_truth` ?
- verify that functions in `ops.py` are the same

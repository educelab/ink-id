# INFO on code... to be filled in later

## @Jack NOTES:
- init():
  - reads in list of volumes & the ground truth
    - if there is a single volume, then you must specify config["multipower"]: False
    - self.volume is of shape [num_volumes, x, y, z]
    - self.groundTruth is of shape [x, y]
  - if config["crop"] == True: then crops based on config[] crop values
  - if config["surface_segmentation"] == True: then open surface segmentation file


- getSamples():
  - grabs samples of size sub-volume from the defined coordinate positions
    - parameters:
      - config dictionary
      - coordinates
      - FLAGS:
        - addNoise
        - augment
        - addJitter
        - includeEdgeSubVolumes

    - returns batch of shape [batch_size, x, y, z, num_volumes]

# TODO:
    # change display step to 100
    # change predict step to 2000
    # change the directory names back
    # change stride to 1
    # change argv:
    #     cropx1
    #     cropx2
    #     cropy1
    #     cropy2
    #     savePredictionPath
    #     experimentType
        # saveModelPath --> uncomment savemodel code
        # scalingFactor

import os
import pdb

os.system("python3 main.py {} {}".format("/home/volcart/volumes/test/MP-single-channel_model/", "multipower-single-channel"))
os.system("python3 main.py {} {}".format("/home/volcart/volumes/test/MP-multi-network_model/", "multipower-multinetwork"))


# crop = [[[477, 819], [72, 1410]], [[1053,1347], [81,1476]], [[1611,1896], [81,1512]]]

# pathCount = 0
# for c in crop:
#     os.system("python3 main.py " + str(c[0][0]) + " " + str(c[0][1]) + " " + str(c[1][0]) + " " + str(c[1][1]) + " true " + paths[pathCount] + " 6")
#     pathCount += 1
#     os.system("python3 main.py " + str(c[0][0]) + " " + str(c[0][1]) + " " + str(c[1][0]) + " " + str(c[1][1]) + " false " + paths[pathCount] + " 1")
#     pathCount += 1
#

# savePath = "/home/volcart/supervised-results/"
# paths = [savePath + "multipower-single-channel/col6/scale1/", savePath + "multipower-single-channel/col6/scale1.8/", savePath + "multipower-single-channel/col6/scale5/",\
#     savePath + "multipower-multi-network/col6/scale1/", savePath + "multipower-multi-network/col6/scale1.8/", savePath + "multipower-multi-network/col6/scale5/"]
# experimentTypes = ["multipower-single-channel", "multipower-multinetwork"]
# scalingFactors = [1, 1.8, 5]
#
# crop = [[456, 825], [66, 1404]]


# command = str(crop[0][0]) + " " + str(crop[0][1]) + " " + str(crop[1][0]) + " " + str(crop[1][1]) + " " + paths[0] + " " + experimentTypes[0] +\
#     " " + paths[0] + " " + str(scalingFactors[0])
#
# os.system("python3 main.py " + command)
#
# exit()
#
# command = str(crop[0][0]) + " " + str(crop[0][1]) + " " + str(crop[1][0]) + " " + str(crop[1][1]) + " " + paths[1] + " " + experimentTypes[0] +\
#     " " + paths[1] + " " + str(scalingFactors[1])
#
# os.system("python3 main.py " + command)
#
# command = str(crop[0][0]) + " " + str(crop[0][1]) + " " + str(crop[1][0]) + " " + str(crop[1][1]) + " " + paths[2] + " " + experimentTypes[0] +\
#     " " + paths[2] + " " + str(scalingFactors[2])
#
# os.system("python3 main.py " + command)


# NOTE: this is the same as the first one commented out
# command = str(crop[0][0]) + " " + str(crop[0][1]) + " " + str(crop[1][0]) + " " + str(crop[1][1]) + " " + paths[0] + " " + experimentTypes[0] +\
#     " " + paths[0] + " " + str(scalingFactors[0])
#
# os.system("python3 main.py " + command)
#
# command = str(crop[0][0]) + " " + str(crop[0][1]) + " " + str(crop[1][0]) + " " + str(crop[1][1]) + " " + paths[3] + " " + experimentTypes[1] +\
#     " " + paths[3] + " " + str(scalingFactors[0])
#
# os.system("python3 main.py " + command)




# command = str(crop[0][0]) + " " + str(crop[0][1]) + " " + str(crop[1][0]) + " " + str(crop[1][1]) + " " + paths[4] + " " + experimentTypes[1] +\
#     " " + paths[4] + " " + str(scalingFactors[1])
#
# os.system("python3 main.py " + command)
#
# command = str(crop[0][0]) + " " + str(crop[0][1]) + " " + str(crop[1][0]) + " " + str(crop[1][1]) + " " + paths[5] + " " + experimentTypes[1] +\
#     " " + paths[5] + " " + str(scalingFactors[2])
#
# os.system("python3 main.py " + command)

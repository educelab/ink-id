import os
import pdb

savePath = "/home/volcart/supervised-results/multipower/"
paths = [savePath + "col6_MP/", savePath + "col6/", savePath + "col5_MP/", savePath + "col5/", savePath + "col4_MP/", savePath + "col4/"]

crop = [[[477, 819], [72, 1410]], [[1053,1347], [81,1476]], [[1611,1896], [81,1512]]]
pathCount = 0
for c in crop:
    os.system("python3 main.py " + str(c[0][0]) + " " + str(c[0][1]) + " " + str(c[1][0]) + " " + str(c[1][1]) + " true " + paths[pathCount] + " 6")
    pathCount += 1
    os.system("python3 main.py " + str(c[0][0]) + " " + str(c[0][1]) + " " + str(c[1][0]) + " " + str(c[1][1]) + " false " + paths[pathCount] + " 1")
    pathCount += 1

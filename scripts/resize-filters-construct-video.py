import cv2
import os
import pdb
import re

rootPath = "/home/volcart/unsupervised-results/CarbonPhantom-Feb2017/Column-6/Dimension-25-numNeurons-32-norm-zNotDownsampled/"
filtersPaths = [rootPath + "1/", rootPath + "2/", rootPath + "3/", rootPath + "4/", rootPath + "5/", rootPath + "6/"]

savePath = "/home/volcart/unsupervised-results/CarbonPhantom-Feb2017/Column-6/Dimension-25-numNeurons-32-norm-zNotDownsampled/resampled/"
count = 0
for f in filtersPaths:
    frameDirs = os.listdir(f)
    frameDirs.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    for n in frameDirs:
        frames = os.listdir(f+n)
        frames.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        for frame in frames:
            im = cv2.imread(f+n+"/"+frame)
            resized = cv2.resize(im, (649, 1479), interpolation = cv2.INTER_CUBIC)

            newFileName = str(count).zfill(5) + ".jpg"
            cv2.imwrite(savePath+newFileName, resized)
            count += 1

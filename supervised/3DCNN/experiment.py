import os

xys = [30,40,50,60,70]
zs = [60,80,100,120,140]
cushions = [10,20,30]

for xy in xys:
    for z in zs:
        for cushion in cushions:
            os.system("python3 main.py {} {} {} 1".format(xy, z, cushion))

import os

xys = [50,70]
zs = [60,140]
cushions = [20]
dropouts = [0.5, 0.7]

for xy in xys:
    for z in zs:
        for cushion in cushions:
            for dropout in dropouts:
                os.system("python3 main.py {} {} {} 1 {}".format(xy, z, cushion, dropout))

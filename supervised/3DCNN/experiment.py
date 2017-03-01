import os

xys = [25, 50]
zs = [60,140]
cushions = [20]
dropouts = [0.5]
paths = ["/home/jack/devel/volcart/small-fragment-data/nudge-0.50%/slices/", "/home/jack/devel/volcart/small-fragment-data/nudge-1.00%/slices/"]

for xy in xys:
    for z in zs:
        for cushion in cushions:
            for dropout in dropouts:
                for path in paths:
                    os.system("python3 main.py {} {} {} 1 {} {}".format(xy, z, cushion, dropout, path))

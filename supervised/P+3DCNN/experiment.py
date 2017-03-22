import os

xys = [50]
zs = [100]
cushions = [20]
dropouts = [0.5]
overlap = 8
paths = [
        "/home/jack/devel/volcart/small-fragment-data/flatfielded-slices/",
        ]

try:
    for xy in xys:
        for z in zs:
            for cushion in cushions:
                for dropout in dropouts:
                    for path in paths:
                        os.system("python3 main.py {} {} {} {} {} {}".format(
                                xy, z, cushion, overlap, dropout, path))

except KeyboardInterrupt:
    # stop everything, instead of just one script
    #TODO make this work
    sys.exit()

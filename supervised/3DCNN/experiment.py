import os

xys = [64]
zs = [64]

paths = [
        "/home/jack/devel/volcart/small-fragment-data/flatfielded-slices/",
        ]

try:
    for xy in xys:
        for z in zs:
            os.system("python3 main.py {} {}".format(
                                xy, z))


except KeyboardInterrupt:
    # stop everything, instead of just one script
    #TODO make this work
    sys.exit()

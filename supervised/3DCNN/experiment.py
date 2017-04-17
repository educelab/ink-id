import os

xys = [64]
zs = [64]
neurons = [2,4,8]
overlap = 2
bounds = [3,0,1,2] # bounds parameters: 0=TOP || 1=RIGHT || 2=BOTTOM || 3=LEFT
paths = [
        #"/home/jack/devel/volcart/small-fragment-data/nudge-4.00%/slices/",
        #"/home/jack/devel/volcart/small-fragment-data/nudge-2.00%/slices/",
        #"/home/jack/devel/volcart/small-fragment-data/nudge-1.50%/slices/",
        #"/home/jack/devel/volcart/small-fragment-data/nudge-1.00%/slices/",
        #"/home/jack/devel/volcart/small-fragment-data/nudge-8.00%/slices/",
        #"/home/jack/devel/volcart/small-fragment-data/nudge-0.50%/slices/".
        "/home/jack/devel/volcart/small-fragment-data/flatfielded-slices/",
        ]

try:
    for xy in xys:
        for z in zs:
            for bound in bounds:
                for path in paths:
                    for neuron in neurons:
                        os.system("python3 main.py {} {} {} {} {}".format(
                                xy, z, bound, neuron, path))


except KeyboardInterrupt:
    # stop everything, instead of just one script
    #TODO make this work
    sys.exit()

import os

xys = [60]
zs = [60]
cushions = [10]
neurons = [4]
overlap = 2
paths = [
        #"/home/jack/devel/volcart/small-fragment-data/nudge-4.00%/slices/",
        #"/home/jack/devel/volcart/small-fragment-data/nudge-2.00%/slices/",
        #"/home/jack/devel/volcart/small-fragment-data/nudge-1.50%/slices/",
        #"/home/jack/devel/volcart/small-fragment-data/nudge-1.00%/slices/",
        #"/home/jack/devel/volcart/small-fragment-data/nudge-8.00%/slices/",
        #"/home/jack/devel/volcart/small-fragment-data/nudge-0.50%/slices/".
        "/home/tfusers/small-fragment-data/flatfielded-slices/",
        ]

neuron = 16
try:
    for xy in xys:
        for z in zs:
            for cushion in cushions:
                for path in paths:
                    while neuron < 100:
                        print("Running experiment with {} neurons".format(neuron))
                        os.system("python3 main.py {} {} {} {} {} {}".format(
                                xy, z, cushion, overlap, neuron, path))
                        neuron += 2


except KeyboardInterrupt:
    # stop everything, instead of just one script
    #TODO make this work
    sys.exit()

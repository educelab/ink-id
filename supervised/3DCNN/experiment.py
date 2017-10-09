import os

neurons = []
neurons.append([16, 8, 4, 2])
lengths = [104,112,120]

for brain in neurons:
    for length in lengths:
        os.system("python3 main.py {} {} {} {} {}".format(brain[0], brain[1], brain[2], brain[3], length))

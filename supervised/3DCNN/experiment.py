import os

neurons = []
neurons.append([16, 8, 4, 2])


for brain in neurons:
    os.system("python3 main.py {} {} {} {}".format(brain[0], brain[1], brain[2], brain[3]))

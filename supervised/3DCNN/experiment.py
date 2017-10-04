import os

neurons = []
neurons.append([16, 8, 4, 2])

squares = [1,3,5,7,9]

for brain in neurons:
    for square in squares:
        os.system("python3 main.py {} {} {} {} {}".format(brain[0], brain[1], brain[2], brain[3], square))

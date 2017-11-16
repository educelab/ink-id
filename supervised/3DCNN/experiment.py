import os


down_then_up = [.5, .33333, .25, .2, .19]
squares = [4,1]

for square in squares:
    os.system("python3 main.py {} {} {}".format(1.0, 1.0, square))

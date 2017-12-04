import os


down_then_up = [.5]

for scale in down_then_up:
    os.system("python3 main.py {}".format(scale))

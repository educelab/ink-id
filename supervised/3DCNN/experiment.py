import os


scale_factors = [.75, .5, .25, .19]

for factor in scale_factors:
    os.system("python3 main.py {}".format(factor))

import os


scale_factors = [.5, .33, .19]

for factor in scale_factors:
    os.system("python3 main.py {}".format(factor))

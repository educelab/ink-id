import os

savePath = "/home/volcart/Desktop/supervisedPredictions/"
for i in range(40,80,10):
    #os.system("mkdir " + savePath + str(i))
    os.system("python3 main.py " + str(i) + " " + str(1))

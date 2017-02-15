import os

savePath = "/home/volcart/Desktop/supervisedPredictions/"
for i in range(75,90,5):
    os.system("mkdir " + savePath + str(i))
    os.system("python main.py " + str(i) + " " + savePath + str(i) + "/ " + str(10))

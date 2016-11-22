'''
inspect-neigh.py
a quickly-written script to invstigate optimal neighborhood sizes
'''


import numpy as np
import tifffile as tiff


def main():
    volume = np.load('volume.npy')
    peaks = np.load('surf-peaks-21000.npy')

    radii = [2,4,8,16]
    lengths = [0,2,4,8]


    display_step = 100

    for rad in radii:
        for length in lengths:
            out = np.zeros((peaks.shape[0], peaks.shape[1]), dtype=np.uint16)
            for i in range(rad, peaks.shape[0]-rad-1):
                for j in range(rad, peaks.shape[1]-rad-1):
                    p = peaks[i][j]
                    all_vects = (volume[i-rad:i+rad+1, j-rad:j+rad+1, p-length:p+length+1])
                    mean = int(np.nan_to_num(np.mean(all_vects)))
                    out[i][j] = mean
                if (i % display_step == 0):
                    print("finished row {}".format(i))
            tiff.imsave("predictions/mean-rad{}-len{}.tif".format(rad,length), out)
            print("saved picture for radius {} length {}".format(rad, length))






if __name__ == "__main__":
    main()

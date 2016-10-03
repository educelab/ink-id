'''
extract-surf-voxels.py
Walk through the fragment, grabbing surface voxels
Output:
    output.txt    all the 3d start points
    output.tiff    a 2D picture with the values at all the start points
'''

import tifffile as tiff
import numpy as np
from scipy.signal import argrelmax,argrelmin
from os import mkdir
import time
import sys


__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"


def main():
    threshes = []
    threshes.append(20000)
    threshes.append(20100)
    threshes.append(20200)
    threshes.append(20300)
    threshes.append(20400)
    threshes.append(20500)

    for thresh in threshes:
        print("beginning threshold {} of {}".format(threshes.index(thresh),len(threshes)))
        extract_at_thresh(thresh)


def extract_at_thresh(a_thresh):
    global NUM_VOX
    global THRESH
    NUM_VOX = 3
    THRESH = a_thresh

    global data_path
    global output_path
    data_path = "/home/jack/devel/ink-id/small-fragment-data"
    output_path = data_path+"/diff-output-"+str(THRESH)

    try:
        mkdir(output_path)
    except Exception:
        print("output directory already exists")

    ref_photo = tiff.imread(data_path+"/registered/aligned-photo-contrast.tif")
    print("loaded data")

    global img_frnt
    global img_back
    num_slices = ref_photo.shape[0]
    slice_length = ref_photo.shape[1]
    output_dims = (num_slices,slice_length)
    img_frnt = np.zeros(output_dims,dtype=np.uint16)
    img_back = np.zeros(output_dims,dtype=np.uint16)

    global end_pts
    global strt_pts
    end_pts = []
    strt_pts = []


    print("initialized output for threshold {}".format(THRESH))


    # # # # # # #
    # MAIN LOOP #
    # # # # # # #
    start_time = time.time()
    for slice_number in range(num_slices):
        # worker
        # extract_surface_for_slice(slice_number)
        extract_width_for_slice(slice_number)

        # progress update
        if(slice_number % (num_slices / 20) == 0):
            pct = float(slice_number) / float(num_slices)
            print("{0} slices complete out of {1} ({2:.2f}%)".format(slice_number,num_slices,pct))


    duration = time.time() - start_time
    print("extraction took {:.2f} seconds ({:.2f} minutes)".format(duration,duration/60))

    print("outputting data for threshold {}".format(THRESH))

    tiff.imsave(output_path+"/front-{}.tif".format(THRESH),img_frnt)
    tiff.imsave(output_path+"/back-{}.tif".format(THRESH),img_back)
    np.savetxt(output_path+"/end-pts-{}.txt".format(THRESH),np.array(end_pts),fmt="%u",delimiter=",")
    np.savetxt(output_path+"/start-pts-{}.txt".format(THRESH),np.array(strt_pts),fmt="%u",delimiter=",")
    np.savetxt(output_path+"/img-back-{}.txt".format(THRESH),img_back,fmt="%u",delimiter=",")
    np.savetxt(output_path+"/img-front-{}.txt".format(THRESH),img_frnt,fmt="%u",delimiter=",")

    print("data outputted for threshold {}".format(THRESH))




def extract_surface_for_slice(a_slice_number):
    i = a_slice_number
    the_slice_name = (data_path+"/vertical_rotated_slices/slice" \
            +"0000"[:4-len(str(i))] + str(i)+".tif")
    skel_slice_name = (output_path+"/skeleton_slices/slice" \
            +"0000"[:4-len(str(i))] + str(i)+".tif")

    the_slice = tiff.imread(the_slice_name)
    skel_slice = np.zeros(the_slice.shape)

    # filter out everything beneath the threshold
    thresh_mat = np.where(the_slice > THRESH,the_slice,0)

    for v in range(the_slice.shape[0]):
        # vect = the_slice[v]
        # filter out everything beneath the threshold
        frag_width = np.where(thresh_mat[v] > 0)
        # find relative maximas
        # vect_peaks = argrelmax(vect)[0]
        if (len(frag_width[0]) > 1):
            start = frag_width[0][0]
            end = frag_width[0][-1]
            skel_slice[v][start] = the_slice[v][start]
            skel_slice[v][end] = the_slice[v][end]
            strt_pts.append((i,v,start))
            end_pts.append((i,v,end))
            img_frnt[i][v] = np.average(the_slice[v][start:start+NUM_VOX])
            img_back[i][v] = np.average(the_slice[v][end-NUM_VOX:end])

    tiff.imsave(skel_slice_name,skel_slice)




def extract_width_for_slice(a_slice_number):
    i = a_slice_number
    the_slice_name = (data_path+"/vertical_rotated_slices/slice"
                      + "0000"[:4-len(str(i))] + str(i)+".tif")
    the_slice = tiff.imread(the_slice_name)
   

    # filter out everything beneath the threshold
    thresh_mat = np.where(the_slice > THRESH, the_slice, 0)

    for v in range(the_slice.shape[0]):
        try:
            thresh_vect = thresh_mat[v]
            vect = the_slice[v]
    
            # find the first peak above the vector
            vect_peaks = argrelmax(thresh_vect)[0]
            speak = vect_peaks[0]
    
            # find the valley next to that peak
            vect_valls = argrelmin(vect[:speak])[0]
            svall = vect_valls[0]
    
            # find the width between the two
            dist = speak - svall
            img_frnt[i][v] = dist

        except Exception:
            # peaks/valleys didn't behave
            pass
    






if __name__ == "__main__":
    main()

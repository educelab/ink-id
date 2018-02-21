'''
extract-surf-voxels.py
Walk through the fragment, grabbing surface voxels using different methods

Output:
    output.txt    3d points representing the edges of the fragment
    output.tif    a 2D picture with the values at all the start points
'''

import tifffile as tiff
import numpy as np
from scipy.signal import argrelmax, argrelmin, resample
from os import mkdir
import time
import warnings
# import pickle as pickle


__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"


def main():
    global data_path
    data_path = "/home/jack/devel/volcart/small-fragment-data"
    warnings.simplefilter('ignore', np.RankWarning)

    ref_photo = tiff.imread(data_path+"/registered/aligned-photo-contrast.tif")
    global num_slices
    global slice_length
    global output_dims
    num_slices = ref_photo.shape[0]
    slice_length = ref_photo.shape[1]
    output_dims = (num_slices, slice_length)

    threshes = []
    threshes.append(21000)
    #threshes.append(21500)

    for thresh in threshes:
        print("beginning threshold {} of {}".format(
                threshes.index(thresh)+1, len(threshes)))
        extract_volume_at_thresh(thresh)


def extract_volume_at_thresh(a_thresh):
    global THRESH
    THRESH = a_thresh

    # the state determines which mode to execute the extraction
    # current options:
    # : surface = first peak value above THRESH,
    # : surface = first valley around THRESH,
    # : surface = sum of values between valley and peak
    # widthState():  # surface = distance between valley and peak
    global state
    #state = resampleState() # surface = resampling
    #state = polyfitState() # surface = line of best fit
    #state = surfState() # surface = first point to cross THRESH of best fit
    state = widthState()

    print("finished intialization for threshold {}".format(THRESH))


    # # # # # # # # #
    # # MAIN LOOP # #
    # # # # # # # # #
    start_time = time.time()
    #new_volume = []
    state.say_my_name()
    for slice_number in range(num_slices):
        # worker
        state.extract_slice(slice_number)

        # progress update
        if(slice_number % (num_slices / 20) == 0):
            pct = float(float(slice_number) / float(num_slices))
            pct = 100 * pct
            print("{0} slices complete out of {1} ({2:.2f}%)".format(
                            slice_number, num_slices, pct))
    # # # # # # # # #
    # # END LOOP  # #
    # # # # # # # # #


    duration = time.time() - start_time
    print("extraction took {:.2f} seconds ({:.2f} minutes)".format(
            duration, duration/60))

    state.save_output()

    print("data outputted for threshold {}".format(THRESH))





'''
States for extraction
'''
class State(object):
    """Base State for extraction modes"""
    def say_my_name(self):
        print("My mode is {}".format(self.mode_name))


class polyfitState(State):
    def __init__(self):
        self.degree = 32
        self.cushion = 16
        self.mode_name = "polyfit"
        self.mode_description = "values from a line of best fit along the surface"
        self.output_path = data_path + "/polyfit-slices-degree{}-cush{}-thresh{}".format(
                self.degree, self.cushion, THRESH)
        self.img_frnt = np.zeros(output_dims, dtype=np.uint16)
        self.surface_points = np.zeros(output_dims, dtype=np.uint16)
        try:
            mkdir(self.output_path)
        except Exception:
            print("output directory \"{}\" may already exist".format(self.output_path))
    
    def extract_slice(self, slice_num):
        image, points = polyfit_for_slice(
                slice_num, degree=self.degree, cushion=self.cushion)
        self.surface_points[slice_num] = points

        tiff.imsave(self.output_path+"/slice{}.tif".format(
            "0000"[:4-len(str(slice_num))] + str(slice_num)),  image)

    def save_output(self):
        tiff.imsave(self.output_path+"/surface.tif", self.surface_points)




class widthState(State):
    def __init__(self):
        self.mode_name = "width"
        self.mode_description = "output dist between surface valley and surface peak"
        self.output_path = data_path + "/width-output-" + str(THRESH)
        self.img_frnt = np.zeros(output_dims, dtype=np.uint16)
        try:
            mkdir(self.output_path)
        except Exception:
            print("output directory \"{}\" may already exist".format(self.output_path))

    def extract_slice(self,slice_num):
        self.img_frnt[slice_num] = extract_width_for_slice(slice_num)

    def save_output(self):
        self.img_frnt = 65535 * ((self.img_frnt - np.min(self.img_frnt)) / (np.max(self.img_frnt) - np.min(self.img_frnt)))
        tiff.imsave(self.output_path+"/front-{}.tif".format(THRESH), self.img_frnt)
        #np.savetxt(self.output_path+"/img-front-{}.txt".format(THRESH),
        #           self.img_frnt, fmt="%u", delimiter=",")


class surfState(State):
    def __init__(self):
        self.mode_name = "surface"
        self.mode_description = "output values at surface peak"
        self.img_frnt = np.zeros(output_dims, dtype=np.uint16)
        self.surface_points = np.zeros(output_dims, dtype=np.uint16)
        self.output_path = data_path + "/surf-output-" + str(THRESH)
        self.skel_path = self.output_path + "/skeleton-slices"
        paths = [self.output_path,self.skel_path]
        for path in paths:
            try:
                mkdir(self.output_path)
            except Exception:
                print("output directory \"{}\" may already exist".format(self.output_path))

    def extract_slice(self,slice_num):
        skel_slice_name = (self.output_path+"/skeleton-slices/slice"
                        + "0000"[:4-len(str(slice_num))] + str(slice_num) + ".tif")
        new_vect, skel_slice, surf_points = extract_surface_for_slice(slice_num)
        self.surface_points[slice_num] = surf_points
        self.img_frnt[slice_num] = new_vect
        #tiff.imsave(skel_slice_name,skel_slice)
        
    def save_output(self): tiff.imsave(self.output_path+"/surface-points-{}.tif".format(
        THRESH), self.surface_points)


class resampleState(State):
    def __init__(self):
        self.new_vol = []
        self.mode_name = "resample"
        self.mode_description = "resample the data so that all slices are constant width"
        self.output_path = data_path + "/resample-output-" + str(THRESH)
        self.skel_path = self.output_path+"/skeleton-slices"
        self.slice_path = self.output_path+"/resampled-slices"
        paths = [self.output_path,self.skel_path,self.slice_path]
        for path in paths:
            try:
                mkdir(path)
            except Exception:
                print("output directory \"{}\" may already exist".format(path))

    def extract_slice(self,slice_num):
        skel_slice_name = (self.skel_path+"/slice"
                        + "0000"[:4-len(str(slice_num))] + str(slice_num) + ".tif")
        new_vect,skel_slice = resample_surface_for_slice(slice_num)
        self.new_vol.append(new_vect)
        tiff.imsave(skel_slice_name,skel_slice)

    def save_output(self):
        # determine maximum length for resizing
        lengths = []
        for s in self.new_vol:
            for i in range(slice_length):
                lengths.append(len(s[i]))
        max_length = max(lengths)

        # resample every slice to max_length and save it
        for i in range(num_slices):
            new_slice = np.zeros((slice_length,max_length), dtype = np.uint16)
            for v in range(slice_length):
                if len(self.new_vol[i][v]) != 0:
                    new_slice[v] = resample(self.new_vol[i][v],max_length)

            slice_name = self.slice_path+"/slice"+"0000"[:4-len(str(i))] + str(i) + ".tif"
            tiff.imsave(slice_name, new_slice)


        



'''
Extraction method implementations
'''

def extract_surface_for_slice(a_slice_number):
    NUM_VOX = 3
    i = a_slice_number
    the_slice_name = (data_path+"/vertical_rotated_slices/slice"
                      + "0000"[:4-len(str(i))] + str(i) + ".tif")

    the_slice = tiff.imread(the_slice_name)
    skel_slice = np.zeros(the_slice.shape)

    # filter out everything beneath the threshold
    thresh_mat = np.where(the_slice > THRESH, the_slice, 0)

    new_vect = [0]*len(the_slice)
    surf_points = np.zeros(len(the_slice), dtype=np.uint16)
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
            new_vect[v] = np.average(the_slice[v][start:start+NUM_VOX])
            surf_points[v] = start

    return new_vect, skel_slice, surf_points


def resample_surface_for_slice(a_slice_number):
    i = a_slice_number
    the_slice_name = (data_path+"/vertical_rotated_slices/slice"
                      + "0000"[:4-len(str(i))] + str(i) + ".tif")

    the_slice = tiff.imread(the_slice_name)
    skel_slice = np.zeros(the_slice.shape, dtype=np.uint16)
    new_vects = [[]]*len(the_slice)

    # filter out everything beneath the threshold
    thresh_mat = np.where(the_slice > THRESH, the_slice, 0)

    for v in range(the_slice.shape[0]):
        try:
            vect = the_slice[v]
            thresh_vect = thresh_mat[v]

            # find peaks above threshold
            peaks = argrelmax(thresh_vect)
            # find the surrounding valleys
            # first valley: the last of all valleys until first peak
            vall1 = argrelmin(vect[:peaks[0][0]])[0][-1]
            # second valley: the first of all valleys after last peak
            vall2 = argrelmin(vect[peaks[0][-1]:])[0][0]
            vall2 += peaks[0][-1] # correct for array trimming

            # show the skeleton
            skel_slice[v][vall1] = the_slice[v][vall1]
            skel_slice[v][vall2] = the_slice[v][vall2]

            # fragment-only vector goes into new vect
            new_vects[v] = vect[vall1:vall2]
        except Exception:
            # not enough data
            pass

    return (new_vects,skel_slice)



def polyfit_for_slice(a_slice_number, degree=32, cushion=2):
    i = a_slice_number
    the_slice_name = (data_path+"/flatfielded-slices/slice"
                      + "0000"[:4-len(str(i))] + str(i)+".tif")
    the_slice = tiff.imread(the_slice_name)
    thresh_mat = np.where(the_slice > THRESH, the_slice, 0)

    # initialize
    return_slice = np.zeros(the_slice.shape, dtype=np.uint16)
    return_points = np.zeros(the_slice.shape[0])
    x_vals = []
    y_vals_front = []
    y_vals_back = []

    # extract the surface peaks
    for v in range(the_slice.shape[0]):
        try:
            thresh_vect = thresh_mat[v]
            # find peaks above threshold
            peaks = argrelmax(thresh_vect)[0]
            front_ind = peaks[0]
            back_ind = peaks[-1]

            if (front_ind > 0 and front_ind < the_slice.shape[1] and 
                    back_ind > 0 and back_ind < the_slice.shape[1] and
                    front_ind != back_ind):
                x_vals.append(v)
                y_vals_front.append(front_ind)
                y_vals_back.append(back_ind)

        except Exception:
            # when no relative maxima exist
            pass


    # create lines of best fit
    if len(x_vals) > 0:
        front_surf_line = np.poly1d(np.polyfit(x_vals, y_vals_front, degree))
        back_surf_line = np.poly1d(np.polyfit(x_vals, y_vals_back, degree))
        for v in range(the_slice.shape[0]):
            approx_front_ind = int(front_surf_line(v))
            approx_back_ind = int(back_surf_line(v))

            if (approx_front_ind < cushion+1 or approx_back_ind > the_slice.shape[1]-cushion-1 or
                    approx_back_ind < cushion+1 or approx_back_ind > the_slice.shape[1]-cushion-1):
                pass
            else:
                start = approx_front_ind - cushion
                end = approx_back_ind + cushion
                return_slice[v, start:end] = the_slice[v, start:end]
                return_points[v] = start


    return return_slice, return_points



def extract_width_for_slice(a_slice_number):
    i = a_slice_number
    the_slice_name = (data_path+"/vertical_rotated_slices/slice"
                      + "0000"[:4-len(str(i))] + str(i)+".tif")
    the_slice = tiff.imread(the_slice_name)
    surface = np.zeros(slice_length)

    # filter out everything beneath the threshold
    thresh_mat = np.where(the_slice > THRESH, the_slice, 0)

    for v in range(slice_length):
        try:
            thresh_vect = thresh_mat[v]
            vect = the_slice[v]

            # find the first peak above the vector
            vect_peaks = argrelmax(thresh_vect)[0]
            speak = vect_peaks[0]

            # find the valley next to that peak
            vect_valls = argrelmin(vect[:speak])[0]
            svall = vect_valls[-1]

            # find the width between the two
            dist = vect[speak] - vect[svall]
            surface[v] = dist

        except Exception:
            # peaks/valleys didn't behave
            pass

    return surface


def extract_slope_for_slice(a_slice_number):
    i = a_slice_number
    the_slice_name = (data_path+"/vertical_rotated_slices/slice"
                      + "0000"[:4-len(str(i))] + str(i)+".tif")
    the_slice = tiff.imread(the_slice_name)

    # filter out everything beneath the threshold
    thresh_mat = np.where(the_slice > THRESH, the_slice, 0)
    slope_vals = np.zeros(slice_length, dtype=np.uint16)
    for v in range(slice_length):
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
            dy = vect[speak] - vect[svall]
            dx = speak - svall
            slope_vals[v] = (dy / dx)

        except Exception:
            # peaks/valleys didn't behave
            pass

    return slope_vals



if __name__ == "__main__":
    main()

'''
LogReg/main.py
initial machine learning skeleton for detecting ink and no-ink points
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"


import tifffile as tiff
import numpy as np
import time
from scipy.signal import argrelmax, argrelmin
#import features
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import confusion_matrix


def main():
    start_time = time.time()

    global THRESH
    global CUT_IN
    global CUT_BACK
    global STRP_RNGE
    global NEIGH_RADIUS
    THRESH = 21000 # the intensity threshold at which to extract the surface
    CUT_IN = 4 # how far beyond the surface point to go
    CUT_BACK = 8 # how far behind the surface point to go
    STRP_RNGE = 64  # the radius of the strip to train on
    NEIGH_RADIUS = 4  # the radius of the strip to train on
    NR = NEIGH_RADIUS

    global data_path
    data_path = "/home/jack/devel/volcart/small-fragment-data"
    ground_truth = tiff.imread(data_path + "/registered/ground-truth-mask-full.tif")
    ground_truth = np.where(ground_truth == 255, 1, 0)

    global num_slices
    global slice_length
    global output_dims
    num_slices = ground_truth.shape[0]
    slice_length = ground_truth.shape[1]
    output_dims = (num_slices, slice_length)

    global volume

    print("extracting features...")
    feature_start = time.time()
    feats = np.load('/home/jack/devel/volcart/output/nudged-ad-hoc-feats-norm-20500.npy')
    feats_list = []
    gt_list = []
    for i in range(num_slices):
        for j in range(slice_length):
            feats_list.append(feats[i,j])
            gt_list.append(ground_truth[i,j])
    print("feature extraction took {:.2f} seconds".format(time.time() - feature_start))


    print("arranging selecting train/test sets...")
    selection_start = time.time()
    
    train_inds = [] # the indeces to use for training
    train_set = []
    train_labels = []

    for ind in range(len(feats_list)):
        # for each fragment sample,
        slice_num, vect_num = ind_to_coord(ind)
        # train on the right half of the image
        if (vect_num % slice_length) < int(slice_length / 2):
            train_inds.append(ind)
            #train_set.append(feats[slice_num][vect_num])
            train_set.append(feats_list[ind])
            train_labels.append(ground_truth[slice_num][vect_num])
    assert (len(train_inds) == len(train_set) == len(train_labels))
    train_set = np.array(train_set)
    '''
    # create a trimmed version for training
    trim_size = 100000
    ink_train_trim = np.random.randint(0,len(ink_inds),size=int(trim_size/2))
    no_ink_train_trim = np.random.randint(0,len(no_ink_inds),size=int(trim_size/2))
    train_set_trim = []
    train_labels_trim = []
    train_inds_trim = []
    for i in ink_train_trim:
        ind = ink_inds[i]
        slice_num = int(ind / slice_length)
        vect_num = ind % slice_length
        train_set_trim.append(feats[slice_num][vect_num])
        train_labels_trim.append(ground_truth[slice_num][vect_num])
        train_inds_trim.append(ind)
    for i in no_ink_train_trim:
        ind = no_ink_inds[i]
        slice_num = int(ind / slice_length)
        vect_num = ind % slice_length
        train_set_trim.append(feats[slice_num][vect_num])
        train_labels_trim.append(ground_truth[slice_num][vect_num])
        train_inds_trim.append(ind)
    '''
    print("training samples: {}".format(len(train_inds)))
    print("random selection took {:.2f} seconds".format(time.time() - selection_start))
    print("there are {} ink samples in training set".format(np.count_nonzero(train_labels)))



    # machine learning training and classification
    train_start = time.time()

    clf = LogisticRegression(C=1e5)
    print("train_set.shape: {}".format(train_set.shape))

    # use for individual feature predictions
    feature = "all"
    print("training classifier for feature {}".format(feature))
    clf.fit(train_set, train_labels)
    print("trained classifier in {:.2f} seconds".format(time.time() - train_start))


    print("predicting samples...")
    predicted_pic = np.zeros(ground_truth.shape, dtype=np.uint16)
    proba_pic = np.zeros(ground_truth.shape, dtype=np.uint16)
 
    preds = clf.predict(feats_list)
    for i in range(num_slices * slice_length):
        (sl,v) = ind_to_coord(i)
        predicted_pic[sl,v] = preds[i]*65535

    probs = clf.predict_proba(feats_list)[:,1]
    for i in range(num_slices * slice_length):
        (sl,v) = ind_to_coord(i)
        proba_pic[sl,v] = probs[i]*65535

    training_pic = np.zeros(ground_truth.shape, dtype=np.uint16)
    for ind in train_inds:
        slice_num, vect_num = ind_to_coord(ind)
        training_pic[slice_num][vect_num] = 65535
    tiff.imsave("/home/jack/devel/volcart/predictions/LogReg-training.tif", training_pic)
    tiff.imsave("/home/jack/devel/volcart/predictions/LogReg-proba-in{}-back{}-neigh{}-feat{}.tif".format(
            CUT_IN, CUT_BACK, NR, feature), proba_pic)
    tiff.imsave("/home/jack/devel/volcart/predictions/LogReg-prediction-in{}-back{}-neigh{}-feat{}.tif".format(
            CUT_IN, CUT_BACK, NR, feature), predicted_pic)

    duration = time.time() - start_time
    print("script took {:.2f} seconds ({:.2f} minutes) to finish".format(duration, duration / 60))



def extract_surface(vect,length):
    to_return = np.zeros(length,dtype=np.uint16)
    thresh_vect = np.where(vect > THRESH, vect, 0)

    # find first peak above threshold
    surf_peak = argrelmax(thresh_vect)[0][0]
    # find the surrounding valleys
    # surface valley: the last of all valleys until first peak
    surf_vall = argrelmin(vect[:surf_peak])[0][-1]

    to_return[surf_vall:surf_peak] = vect[surf_vall:surf_peak]
    return to_return



def extract_surf_vall(vect):
    thresh_vect = np.where(vect > THRESH, vect, 0)

    # find first peak above threshold
    surf_peak = argrelmax(thresh_vect)[0][0]
    # find the surrounding valleys
    # surface valley: the last of all valleys until first peak
    surf_vall = argrelmin(vect[:surf_peak])[0][-1]

    return surf_vall



def extract_surf_peak(vect):
    thresh_vect = np.where(vect > THRESH, vect, 0)

    # find first peak above threshold
    surf_peak = argrelmax(thresh_vect)[0][0]

    return surf_peak



def ind_to_coord(an_ind):
    slice_num = int(an_ind / slice_length)
    vect_num = an_ind % slice_length
    return (slice_num, vect_num)


if __name__ == "__main__":
    main()

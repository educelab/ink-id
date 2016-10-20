'''
learn-ink.py
machine learning skeleton for detecting ink and no-ink points
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"


import tifffile as tiff
import numpy as np
import time
from scipy.signal import argrelmax, argrelmin
import features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def main():
    start_time = time.time()

    global THRESH
    global CUT_IN
    global CUT_BACK
    global STRP_RNGE
    global NEIGH_RADIUS
    THRESH = 21000 # the intensity threshold at which to extract the surface
    CUT_IN = 4 # how far beyond the surface point to go
    CUT_BACK = 16 # how far behind the surface point to go
    STRP_RNGE = 64  # the radius of the strip to train on
    NEIGH_RADIUS = 4  # the radius of the strip to train on
    NR = NEIGH_RADIUS

    global data_path
    data_path = "/home/jack/devel/ink-id/small-fragment-data"
    ground_truth = tiff.imread(data_path + "/registered/ground-truth-mask.tif")
    ground_truth = np.where(ground_truth == 255, 1, 0)

    global num_slices
    global slice_length
    global output_dims
    num_slices = ground_truth.shape[0]
    slice_length = ground_truth.shape[1]
    output_dims = (num_slices, slice_length)

    global volume
    #surf_pts_filename = "surf-valls-{}".format(THRESH)
    surf_pts_filename = "surf-peaks-{}".format(THRESH)
    volume_filename = "volume-{}".format(THRESH)
    try:
        surf_pts = np.load(surf_pts_filename+".npy")
        volume = np.load(volume_filename+".npy")
        print("loaded surface points / volume from file")
    except Exception:
        print("failed to load surface points / volume from file")
        print("extracting surface for threshold {}".format(THRESH))

        volume = np.zeros((num_slices, slice_length, 286), dtype=np.uint16)
        surf_pts = np.zeros(output_dims, dtype=np.uint16)
        for sl in range(num_slices):
            slice_name = (data_path+"/vertical_rotated_slices/slice"
                      + "0000"[:4-len(str(sl))] + str(sl) + ".tif")
            current_slice = np.array(tiff.imread(slice_name))
            vect_length = current_slice.shape[1]

            for v in range(slice_length):
                volume[sl][v] = np.zeros(vect_length,dtype=np.uint16)
                try:
                    surf_pts[sl][v] = extract_surf_peak(current_slice[v])
                    volume[sl][v] = current_slice[v]
                except Exception:
                    #print(inst.args)
                    # no definitive peak/valley pair
                    pass
            #print("finished slice number {}, {} extractions and {} fails".format(sl, good, bad))
        np.save(surf_pts_filename, surf_pts)
        np.save(volume_filename, volume)
        print("saved surf_pts / volume to {} / {}".format(surf_pts_filename, volume_filename))


    print("finding samples on the fragment")
    ink_inds = []
    no_ink_inds = []
    for i in range(num_slices):
        for j in range(slice_length):
            # only train where everything in the neighborhood is ink
            gt_flat = ground_truth[i-NR:i+NR+1,j-NR:j+NR+1].flatten()
            needed_gt = (2*NR+1)*(2*NR+1)
            if np.count_nonzero(gt_flat) == needed_gt:
                ink_inds.append(i*slice_length + j)
                #print("gt_flat: {}".format(gt_flat))
                #print("needed_gt: {}".format(needed_gt))
            elif np.count_nonzero(np.where(volume[i][j] > THRESH, volume[i][j], 0)) > 12:
                no_ink_inds.append(i*slice_length + j)
    fragment_inds = ink_inds + no_ink_inds
    print("total samples on fragment: {}".format(len(fragment_inds)))
    print("total ink samples: {}".format(len(ink_inds)))
    print("total no-ink samples: {}".format(len(no_ink_inds)))


    print("extracting features")
    feature_start = time.time()
    feats = features.extract_for_vol(volume, surf_pts, CUT_IN, CUT_BACK, NEIGH_RADIUS, THRESH)
    feats_list = []
    gt_list = []
    for i in range(num_slices):
        for j in range(slice_length):
            feats_list.append(feats[i,j])
            gt_list.append(ground_truth[i,j])
    # extract features for all vectors in the volume
    print("feature extraction took {:.2f} seconds".format(time.time() - feature_start))
    k = 2
    print("selecting {} best features".format(k))
    feats_list = SelectKBest(chi2, k=k).fit_transform(feats_list,gt_list)


    print("randomly selecting train/test sets")
    selection_start = time.time()
    
    # randomly divide train and test data
    total_fragment_samples = len(fragment_inds)
    assignments = np.random.randint(0,10, size=total_fragment_samples)
    test_nums = np.random.randint(0,10,size=2)
    train_inds = [] # the indeces to use for training
    test_inds = [] # the indeces to use for testing
    train_set = []
    test_set = []
    train_labels = []
    test_labels = []

    for ind in range(total_fragment_samples):
        # for each fragment sample,
        # find its index
        i = fragment_inds[ind]
        slice_num, vect_num = ind_to_coord(i)
        if assignments[ind] in test_nums:
            test_inds.append(i)
            #test_set.append(feats[slice_num][vect_num])
            test_set.append(feats_list[i])
            test_labels.append(ground_truth[slice_num][vect_num])
        # train on the right half of the image
        elif (vect_num % slice_length) < int(slice_length / 2):
            train_inds.append(i)
            #train_set.append(feats[slice_num][vect_num])
            train_set.append(feats_list[i])
            train_labels.append(ground_truth[slice_num][vect_num])
    assert (len(train_inds) == len(train_set) == len(train_labels))
    assert (len(test_inds) == len(test_set) == len(test_labels))
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
    s_clf = LogisticRegression(C=1e5)


    print("training classifier")
    clf.fit(train_set, train_labels)
    print("trained classifier in {:.2f} seconds".format(time.time() - train_start))
    print("score on test set: {}".format(clf.score(test_set,test_labels)))
    predicted = clf.predict(test_set)
    print("confusion matrix for test set: {}".format(confusion_matrix(test_labels,predicted)))

    print("predicting samples")
    pred_start = time.time()
    predicted_pic = np.zeros(ground_truth.shape, dtype=np.uint16)
    strip_predicted_pic = np.zeros(ground_truth.shape, dtype=np.uint16)

    train_inds = np.array(train_inds)
    print("train_inds shape: {}".format(train_inds.shape))
 
    preds = clf.predict_proba(feats_list)[:,1]
    for i in range(num_slices * slice_length):
        (sl,v) = ind_to_coord(i)
        predicted_pic[sl,v] = preds[i]*65535


    # localized training
    strip_length = STRP_RNGE * slice_length
    for strip in range(0, int(num_slices / STRP_RNGE)):
        min_ind = (strip * strip_length)
        max_ind = ((strip+1) * strip_length)
        #min_slc = int(min_ind / slice_length)
        #max_slc = int(max_ind / slice_length)

        train_strip_inds = np.where(np.logical_and(train_inds>=min_ind, train_inds <=max_ind))[0]
        #train_strip_set = [train_set[ind] for ind in train_strip_inds]
        #train_strip_labels = [train_labels[ind] for ind in train_strip_inds]
        train_strip_set = []
        train_strip_labels = []
        for ind in train_strip_inds:
            (sl, v) = ind_to_coord(train_inds[ind])
            train_strip_set.append(feats[sl,v])
            train_strip_labels.append(ground_truth[sl,v])
        num_inks = np.count_nonzero(train_strip_labels)
        num_no_inks = len(train_strip_labels) - num_inks
        if(num_inks > 20 and num_no_inks > 20):
            print("training strip {}, min: {}, max: {}".format(strip, min_ind, max_ind))
            s_clf.fit(train_strip_set, train_strip_labels)
            for ind in range(min_ind, max_ind):
                (sl, v) = ind_to_coord(ind)
                vect_feats = feats[sl,v]
                pred = (s_clf.predict_proba([vect_feats]) * 65535)[:,1][0]
                strip_predicted_pic[sl, v] = pred

    tiff.imsave("predictions/prediction-in{}-back{}-neigh{}.tif".format(CUT_IN,CUT_BACK,NR), predicted_pic)
    tiff.imsave("predictions/strip{}-prediction-in{}-back{}-neigh{}.tif".format(
        STRP_RNGE,CUT_IN,CUT_BACK,NR), strip_predicted_pic)
    print("predicted all samples in {} seconds".format(time.time() - pred_start))

    training_pic = np.zeros(ground_truth.shape, dtype=np.uint16)
    for ind in train_inds:
        slice_num, vect_num = ind_to_coord(ind)
        training_pic[slice_num][vect_num] = 65535
    tiff.imsave("predictions/training.tif", training_pic)

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

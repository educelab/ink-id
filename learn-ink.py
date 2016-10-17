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
#from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


def main():
    start_time = time.time()

    global THRESH
    THRESH = 21000

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
    surf_pts_filename = "surf-pts-{}".format(THRESH)
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
                    surf_pts[sl][v] = extract_surf_pt(current_slice[v])
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
            if ground_truth[i][j]:
                ink_inds.append(i*slice_length + j)
            elif np.count_nonzero(volume[i][j]) > 0:
                no_ink_inds.append(i*slice_length + j)
    fragment_inds = ink_inds + no_ink_inds
    print("total samples on fragment: {}".format(len(fragment_inds)))
    print("total ink samples: {}".format(len(ink_inds)))
    print("total no-ink samples: {}".format(len(no_ink_inds)))


    print("extracting features")
    feature_start = time.time()
    # extract features for all vectors in the volume
    feats = features.extract_for_vol(volume, surf_pts, 32)
    print("feature extraction took {:.2f} seconds".format(time.time() - feature_start))


    print("randomly selecting train/test sets")
    selection_start = time.time()
    
    # randomly divide train and test data
    total_fragment_samples = len(fragment_inds)
    assignments = np.random.randint(0,10, size=num_slices*slice_length)
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
        if assignments[i] in test_nums:
            test_inds.append(i)
            test_set.append(feats[slice_num][vect_num])
            test_labels.append(ground_truth[slice_num][vect_num])
        elif (vect_num % slice_length) > int(slice_length / 2):
            train_inds.append(i)
            train_set.append(feats[slice_num][vect_num])
            train_labels.append(ground_truth[slice_num][vect_num])
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

    ink_count = 0
    train_inds = np.array(train_inds)
    print("train_inds shape: {}".format(train_inds.shape))
   
    RNGE = 16
    for sl in range(num_slices - RNGE):
        pred = clf.decision_function(feats[sl])
        predicted_pic[sl] = np.where(pred > 0, pred*65535, 0)
        
        min_ind = (sl * slice_length)
        max_ind = ((sl+1+RNGE) * slice_length)

        train = np.where(np.logical_and(train_inds>=min_ind, train_inds <=max_ind))[0]
        train_slice_set = [train_set[ind] for ind in train]
        train_slice_labels = [train_labels[ind] for ind in train]

        if(np.count_nonzero(train_slice_labels)):
            s_clf.fit(train_slice_set, train_slice_labels)
            pred = s_clf.decision_function(feats[sl])
            ink_count += np.count_nonzero(pred)
            predicted_pic[sl] = np.where(pred > 0, pred*65535, 0)
        
        if (sl % (num_slices / 10) == 0):
            print("finished {:.2f}% of predictions ({:.2f} seconds)".format(
                100*(sl / num_slices), time.time()-pred_start))
    tiff.imsave("prediction.tif", predicted_pic)
    print("predicted all samples in {} seconds".format(time.time() - pred_start))
    print("there are {} predicted ink points in the output".format(ink_count))

    training_pic = np.zeros(ground_truth.shape, dtype=np.uint16)
    for ind in train_inds:
        slice_num, vect_num = ind_to_coord(ind)
        training_pic[slice_num][vect_num] = 65535
    tiff.imsave("training.tif", training_pic)

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


def extract_surf_pt(vect):
    thresh_vect = np.where(vect > THRESH, vect, 0)

    # find first peak above threshold
    surf_peak = argrelmax(thresh_vect)[0][0]
    # find the surrounding valleys
    # surface valley: the last of all valleys until first peak
    surf_vall = argrelmin(vect[:surf_peak])[0][-1]

    return surf_vall


def ind_to_coord(an_ind):
    slice_num = int(an_ind / slice_length)
    vect_num = an_ind % slice_length
    return (slice_num, vect_num)


if __name__ == "__main__":
    main()

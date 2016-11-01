'''
features.py
feature extraction for ink/no-ink vectors
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"

import numpy as np
from sklearn import preprocessing


def extract_for_list(vect_list, neigh_inds, CUT_IN, CUT_BACK, NEIGH_RADIUS):
    num_features = 10
    features = np.zeros((len(vect_list), num_features), dtype=np.float64)

    features_filename = "feat_cache/l-features-in{}-back{}-neigh{}".format(
            num_features, CUT_IN, CUT_BACK, NEIGH_RADIUS)

    try:
        features = np.load(features_filename+".npy")
        print("loaded features from file")
        return features
    except Exception:
        pass


    # vector average
    features[:,0] = (np.mean(vect_list, axis=1))

    # vector integral
    features[:,1] = (np.sum(vect_list, axis=1))

    # vector min
    features[:,2] = (np.min(vect_list, axis=1))

    # vector max
    features[:,3] = (np.max(vect_list, axis=1))

    # vector stdev
    features[:,4] = (np.std(vect_list, axis=1))

    print("extracted single-vector features")

    for i in range(len(neigh_inds)):
        current_neigh = neigh_inds[i,np.nonzero(neigh_inds[i])]

        # neighborhood average
        features[i,5] = np.mean(np.array([features[ind,0] for ind in current_neigh]))

        # neighborhood integral
        features[i,6] = np.sum(np.array([features[ind,1] for ind in current_neigh]))

        # neighborhood min
        features[i,7] = np.min(np.array([features[ind,2] for ind in current_neigh]))

        # neighborhood max
        features[i,8] = np.max(np.array([features[ind,3] for ind in current_neigh]))

        # neighborhood stdev
        features[i,9] = np.std(np.array([vect_list[ind] for ind in current_neigh]).flatten())

    print("extracted neighborhood features")

    features = preprocessing.normalize(features, axis=1)
    np.save(features_filename, features)
    return features


def extract_for_vol(volume, surf_pts, CUT_IN, CUT_BACK, NEIGH_RADIUS, THRESH):
    USE_SINGLE = True
    USE_NEIGH = True

    output_dims = (volume.shape[0], volume.shape[1])

    features = []

    features_filename = "feat_cache/features-in{}-back{}-neigh{}".format(
            CUT_IN, CUT_BACK, NEIGH_RADIUS)
    try:
        features = np.load(features_filename+".npy")
        print("loaded features from file")
        return features
    except Exception:
        pass

    # single-vector features calculate
    depth_vals = np.zeros(output_dims,dtype=np.float64)
    avg_vals = np.zeros(output_dims,dtype=np.float64)
    sum_vals = np.zeros(output_dims,dtype=np.float64)
    min_vals = np.zeros(output_dims,dtype=np.float64)
    max_vals = np.zeros(output_dims,dtype=np.float64)
    std_vals = np.zeros(output_dims,dtype=np.float64)
    for i in range(volume.shape[0]):
        for j in range(volume.shape[1]):
            srf = surf_pts[i,j] - CUT_BACK
            if srf < 0: srf = 0
            cut = srf + CUT_IN
            #print("srf = {}, cut = {}".format(srf,cut))
            tmp_vect = volume[i,j][srf:cut]
            #print(tmp_vect)

            depth = np.count_nonzero(np.where(tmp_vect > THRESH, tmp_vect, 0))
            depth_vals[i,j] = depth

            avg_vals[i,j] = np.mean(tmp_vect)
            sum_vals[i,j] = np.sum(tmp_vect)
            min_vals[i,j] = np.min(tmp_vect)
            max_vals[i,j] = np.max(tmp_vect)
            std_vals[i,j] = np.std(tmp_vect)

    if USE_SINGLE:
        features.append(preprocessing.normalize(np.nan_to_num(sum_vals)))
        features.append(preprocessing.normalize(depth_vals))
        features.append(preprocessing.normalize(avg_vals))
        features.append(preprocessing.normalize(np.nan_to_num(min_vals)))
        features.append(preprocessing.normalize(np.nan_to_num(max_vals)))
        features.append(preprocessing.normalize(np.nan_to_num(std_vals)))

    print("single-vector features: {}".format(len(features)))


    # neighborhood features
    NR = NEIGH_RADIUS
    mean_neigh = np.zeros(output_dims, dtype=np.float64)
    sum_neigh = np.zeros(output_dims, dtype=np.float64)
    surf_stdev_neigh = np.zeros(output_dims, dtype=np.float64)
    depth_neigh = np.zeros(output_dims, dtype=np.float64)

    if USE_NEIGH:
        for i in range(NR, volume.shape[0]-NR):
            for j in range(NR, volume.shape[1]-NR):
                srf = surf_pts[i,j] - CUT_BACK
                if srf < 0: srf = 0
                cut = srf + CUT_IN
                neigh_vects = volume[i-NR:i+NR+1, j-NR:j+NR+1, srf:cut]
                mean_neigh[i,j] = np.mean(neigh_vects.flatten()).astype(np.float64)
                sum_neigh[i,j] = np.sum(neigh_vects.flatten()).astype(np.float64)
                surf_stdev_neigh[i,j] = np.std(neigh_vects.flatten()).astype(np.float64)
                depth_neigh[i,j] = np.count_nonzero(
                        np.where(neigh_vects < THRESH, neigh_vects, 0).flatten())
        features.append(preprocessing.normalize(np.nan_to_num(mean_neigh)))
        features.append(preprocessing.normalize(np.nan_to_num(sum_neigh)))
        features.append(preprocessing.normalize(np.nan_to_num(surf_stdev_neigh)))
        features.append(preprocessing.normalize(np.nan_to_num(depth_neigh)))


    features = np.array(features)
    print("total features: {}".format(len(features)))
    features = np.swapaxes(features,1,2)
    features = np.swapaxes(features,0,2)

    np.save(features_filename, features)
    return features





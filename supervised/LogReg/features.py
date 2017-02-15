'''
features.py
feature extraction for ink/no-ink vectors
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"

import numpy as np
import tifffile as tiff
from sklearn import preprocessing
from scipy.signal import argrelmax, argrelmin


def main():
    """Run feature extraction ad-hoc"""
    print("Initializing...")
    folder = '/home/jack/devel/ink-id/small-fragment-data/'
    vol_front = np.load(folder+'volume-increase797-neigh4-scale2.npy')
    vol_flip = np.flip(vol_front, axis=2)
    vol = np.concatenate((vol_front, vol_flip), axis=1)
    print("Created reversed volume with shape {}".format(vol.shape))
    surf_pts = np.zeros((vol.shape[0], vol.shape[1]), dtype=np.int)

    maxes = np.max(vol, axis=2)
    frag_mask = np.zeros(maxes.shape, dtype=np.uint16)
    surf_pts = np.zeros(maxes.shape, dtype=np.int)
    surf_vals = np.zeros(maxes.shape, dtype=np.int)
    surf_peaks = np.zeros(maxes.shape, dtype=np.int)
    #surf_pts_reverse = np.zeros(maxes.shape, dtype=np.int)

    n = 2 # neighborhood to verify
    thresh = 20500
    CUT_IN = 8
    CUT_BACK = 8

    print("Extracting surface...")
    errors = 0
    successes = 0
    for i in range(n, vol.shape[0]-n-1):
        for j in range(n, vol.shape[1]-n-1):
            # if points in the point's neighborhood exceed the threshold
            if np.min(maxes[i-n:i+n+1, j-n:j+n+1]) > thresh:
                # mark it as a point on the fragment
                frag_mask[i,j] = 1
                try:
                    above = np.where(vol[i,j] > thresh)[0]
                    valley = argrelmin(vol[i,j,:above[0]])[0][-1] # the 'entry' trough 
                    peak = valley + argrelmax(vol[i,j,valley:])[0][0]
                    surface = int((peak + valley) / 2)
                    
                    surf_pts[i,j] = surface
                    surf_peaks[i,j] = peak
                    surf_vals[i,j] = valley 
                    successes += 1
                except Exception:
                    # no definitive peak/valley surface
                    errors += 1
        if (i % int(vol.shape[0] / 10) == 0):
            print('{}% done'.format(int((i / vol.shape[0]) * 100)))
    print("Finished extracting surface with {} no-surface points, {} surface points".format(errors, successes))
    np.save('/home/jack/devel/ink-id/output/ad-hoc-surf-{}'.format(thresh), surf_pts)
    np.save('/home/jack/devel/ink-id/output/ad-hoc-peaks-{}'.format(thresh), surf_peaks)
    np.save('/home/jack/devel/ink-id/output/ad-hoc-valls-{}'.format(thresh), surf_vals)


    feats = extract_for_volume(vol, surf_pts, surf_peaks, surf_vals, CUT_IN, CUT_BACK, thresh)
    np.save('/home/jack/devel/ink-id/output/nudged-ad-hoc-feats-raw-{}'.format(thresh), feats)
    for i in range(feats.shape[2]):
        feats[:,:,i] = (feats[:,:,i] - np.min(feats[:,:,i])) / np.max(feats[:,:,i])

    np.save('/home/jack/devel/ink-id/output/nudged-ad-hoc-feats-norm-{}'.format(thresh), feats)
    for i in range(feats.shape[2]):
        outpic = feats[:,:,i] * 65535
        outpictif = outpic.astype(np.uint16)
        tiff.imsave('/home/jack/devel/ink-id/output/nudged-feat{}-{}'.format(i,thresh), outpictif)






def extract_for_volume(volume, surf_pts, surf_peaks, surf_valls, CUT_IN, CUT_BACK, THRESH):
    num_features = 15
    output_dims = (volume.shape[0], volume.shape[1])
    features = np.zeros((volume.shape[0], volume.shape[1], num_features), dtype=np.float64)

    # fixed-window features
    surf_inds = np.zeros(output_dims, dtype=np.float64)
    vall_inds = np.zeros(output_dims, dtype=np.float64)
    peak_inds = np.zeros(output_dims, dtype=np.float64)
    surf_vals = np.zeros(output_dims, dtype=np.float64)
    vall_vals = np.zeros(output_dims, dtype=np.float64)
    peak_vals = np.zeros(output_dims, dtype=np.float64)
    min_vals = np.zeros(output_dims, dtype=np.float64)
    max_vals = np.zeros(output_dims, dtype=np.float64)
    avg_vals = np.zeros(output_dims, dtype=np.float64)
    sum_vals = np.zeros(output_dims, dtype=np.float64)
    std_vals = np.zeros(output_dims, dtype=np.float64)

    # d_ynamic features: changes depending on peak and valley location
    d_width_vals = np.zeros(output_dims, dtype=np.float64)
    d_avg_vals = np.zeros(output_dims, dtype=np.float64)
    d_sum_vals = np.zeros(output_dims, dtype=np.float64)
    d_std_vals = np.zeros(output_dims, dtype=np.float64)

    count = 0
    print("Extracting features...")
    for i in range(volume.shape[0]):
        for j in range(volume.shape[1]):
            surface = surf_pts[i,j]
            valley = surf_valls[i,j]
            peak = surf_peaks[i,j]

            if peak > 0 and valley > 0:
                try:
                    vector = volume[i,j, surface-CUT_BACK:surface+CUT_IN]
                    d_vector = volume[i,j, valley:peak]

                    surf_inds[i,j] = surface
                    vall_inds[i,j] = valley
                    peak_inds[i,j] = peak
                    surf_vals[i,j] = volume[i,j,surface]
                    vall_vals[i,j] = volume[i,j,valley]
                    peak_vals[i,j] = volume[i,j,peak]
                    min_vals[i,j] = np.min(vector)
                    max_vals[i,j] = np.max(vector)
                    avg_vals[i,j] = np.mean(vector)
                    sum_vals[i,j] = np.sum(vector)
                    std_vals[i,j] = np.std(vector)

                    # d_ynamic features
                    d_width_vals[i,j] = peak - valley
                    d_avg_vals[i,j] = np.mean(d_vector)
                    d_sum_vals[i,j] = np.sum(d_vector)
                    d_std_vals[i,j] = np.std(d_vector)
                    count += 1
                except Exception:
                    # catches "zero-size array to reduction operation" error
                    pass

        # progress update
        if (i % (int(volume.shape[0] / 10))) == 0:
            print("{}% done".format( int(i / volume.shape[0] * 100)))

    print("Extracted at {} points".format(count))
    features[:,:,0] = surf_inds 
    features[:,:,1] = vall_inds
    features[:,:,2] = peak_inds
    features[:,:,3] = surf_vals 
    features[:,:,4] = vall_vals
    features[:,:,5] = peak_vals
    features[:,:,6] = min_vals
    features[:,:,7] = max_vals
    features[:,:,8] = avg_vals
    features[:,:,9] = sum_vals
    features[:,:,10] = std_vals
    features[:,:,11] = d_width_vals
    features[:,:,12] = d_avg_vals
    features[:,:,13] = d_sum_vals
    features[:,:,14] = d_std_vals

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



def extract_for_list(vect_list, neigh_inds, CUT_IN, CUT_BACK, NEIGH_RADIUS):
    num_features = 12
    features = np.zeros((len(vect_list), num_features), dtype=np.float32)

    features_filename = "feat_cache/l-{}features-in{}-back{}-neigh{}".format(
            num_features, CUT_IN, CUT_BACK, NEIGH_RADIUS)

    try:
        features = np.load(features_filename+".npy")
        print("loaded features from file")
        return features
    except Exception:
        pass


    # vector average
    features[:,0] = np.mean(vect_list, axis=1)

    # vector integral
    features[:,1] = (np.sum(vect_list, axis=1))

    # vector min
    features[:,2] = (np.min(vect_list, axis=1))

    # vector max
    features[:,3] = (np.max(vect_list, axis=1))

    # vector stdev
    features[:,4] = (np.std(vect_list, axis=1))

    # average vector slope 
    features[:,5] = np.mean(np.gradient(vect_list, axis=1), axis=1)

    for i in range(len(neigh_inds)):
        current_neigh = neigh_inds[i,np.nonzero(neigh_inds[i])]

        # neighborhood average
        features[i,6] = np.mean(np.array([features[ind,0] for ind in current_neigh]))

        # neighborhood integral
        features[i,7] = np.sum(np.array([features[ind,1] for ind in current_neigh]))

        # neighborhood min
        features[i,8] = np.min(np.array([features[ind,2] for ind in current_neigh]))

        # neighborhood max
        features[i,9] = np.max(np.array([features[ind,3] for ind in current_neigh]))

        # neighborhood stdev
        features[i,10] = np.std(np.array([vect_list[ind] for ind in current_neigh]).flatten())

        # neighborhood average slope 
        features[i,11] = np.mean(np.array([vect_list[ind,5] for ind in current_neigh]))

    print("extracted neighborhood features")

    for f in range(features.shape[1]):
        f_min = np.min(features[:,f])
        f_max = np.max(features[:,f])
        features[:,f] = (features[:,f] - f_min) / (f_max - f_min)
    np.save(features_filename, features)
    return features


if __name__ == "__main__":
    main()


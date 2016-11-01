'''
learnink.py
machine learning skeleton for detecting ink and no-ink points
refactored and refined for tensorflow implementation
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"


import tifffile as tiff
import numpy as np
import time
from scipy.signal import argrelmax, argrelmin
import features
import tensorflow as tf
from sklearn.linear_model import LogisticRegression 
from sklearn import svm
#from sklearn.metrics import confusion_matrix


def main():
    start_time = time.time()

    print("initializing data")
    global THRESH
    global MIN
    global CUT_IN
    global CUT_BACK
    global STRP_RNGE
    global NEIGH_RADIUS
    global NR
    global neigh_count
    THRESH = 21000 # the intensity threshold at which to extract the surface
    MIN = 2 # the minimum number of points above THRESH to be considered on the fragment
    CUT_IN = 4 # how far beyond the surface point to go
    CUT_BACK = 16 # how far behind the surface point to go
    STRP_RNGE = 8  # the radius of the strip to train on
    NEIGH_RADIUS = 2  # the radius of the strip to train on
    NR = NEIGH_RADIUS
    neigh_count = (2*NR+1)*(2*NR+1)

    global data_path
    global ground_truth
    data_path = "/home/jack/devel/ink-id/small-fragment-data"
    ground_truth = tiff.imread(data_path + "/registered/ground-truth-mask-full.tif")
    ground_truth = np.where(ground_truth == 255, 1, 0)

    global volume
    global surf_pts
    volume = retrieve_volume()
    surf_pts = retrieve_surf_pts(volume)

    global num_slices
    global slice_length
    global vol_depth
    global output_dims
    num_slices = ground_truth.shape[0]
    slice_length = ground_truth.shape[1]
    vol_depth = volume.shape[2]
    output_dims = (num_slices, slice_length)

    print("unraveling volume into 2D array")
    global vol_list # an ordered list of all the vectors in the volume
    vol_list = np.zeros((num_slices*slice_length, vol_depth), dtype=np.int)
    truth_list = np.zeros((num_slices*slice_length), dtype=np.int)
    for sl in range(num_slices):
        for v in range(slice_length):
            v_ind = coord_to_ind(sl,v)
            vol_list[v_ind] = volume[sl][v]
            truth_list[v_ind] = ground_truth[sl][v]
    print("unravelled volume")


    print("finding points on fragment")
    frag_inds = find_frag_inds(THRESH, MIN)
    frag_truth = [truth_list[i] for i in frag_inds]
    print("found {} points on fragment, {} ink".format(
        len(frag_inds), np.count_nonzero(frag_truth)))


    print("arranging input into neighborhoods")
    vect_list, neigh_inds = make_feature_input()
    # create a list of all samples on the fragment
    #frag_truth = [(1,0)]*len(ink_coords) + [(0,1)]*len(no_ink_coords)
    print("arranged input")


    print("extracting features")
    all_feats = features.extract_for_list(vect_list, neigh_inds, CUT_IN, CUT_BACK, NR)


    # make train and test
    n = len(frag_inds)
    np.random.shuffle(frag_inds)
    train_inds = np.array(frag_inds[:int(.6*n)])
    #test_inds = inds[int(.6*n):int(.8*n)]
    #val_inds = inds[int(.8*n):]

    train_x = np.zeros((len(train_inds), all_feats.shape[1]))
    train_y = np.zeros((len(train_inds)))
    #train_y = np.zeros((len(train_inds),2))
    for i in range(len(train_inds)):
        tmp_ind = train_inds[i]
        train_x[i] = all_feats[tmp_ind]
        train_y[i] = truth_list[tmp_ind]
    print("train_x: {}\n".format(train_x[:10]))
    print("train_y: {}\n".format(train_y[:10]))

    print("train_y has {} nonzeros".format(np.count_nonzero(train_y)))
    make_train_truth_pic(train_inds, train_y)
    #test_x = [feats[index] for index in test_inds]
    #test_y = [frag_truth[index] for index in test_inds]
    #preds = learn_tf(train_x, train_y, test_x, test_y, feats)
    preds = learn_lr(train_x, train_y, all_feats)
    make_full_pred_pic(preds)


    '''
    # use this loop for single-feature training
    for i in range(len(feats[0])):
        tmp_x = train_x[:,i].reshape(-1,1)
        tmp_feats = feats[:,i].reshape(-1,1)
        preds = learn_lr(tmp_x, train_y, tmp_feats)
        make_pred_pic(frag_coords, preds, str(i))
    '''
    # use this loop for strip training
    preds = np.zeros(len(all_feats), dtype=np.float32)
    strip_length = STRP_RNGE * slice_length
    for strip in range(0, int(num_slices / STRP_RNGE)):
        min_ind = (strip * strip_length)
        max_ind = ((strip+1) * strip_length)
        train_strip_inds = np.where(np.logical_and(train_inds>=min_ind, train_inds<=max_ind))[0]

        tmp_x = [train_x[ind] for ind in train_strip_inds]
        tmp_y = [train_y[ind] for ind in train_strip_inds]
        tmp_feats = all_feats[min_ind:max_ind]

        num_inks = np.count_nonzero(tmp_y)
        num_no_inks = len(tmp_y) - num_inks
        if(num_inks > 20 and num_no_inks > 20):
            preds[min_ind:max_ind] = learn_lr(tmp_x, tmp_y, tmp_feats)
    make_full_pred_pic(preds,strip='-strip{}'.format(STRP_RNGE))
 

    make_train_pic(train_inds)
    print("full script took {:.2f} seconds".format(time.time() - start_time))



def find_frag_inds(THRESH, MIN):
    ink_inds = []
    no_ink_inds = []
 
    for i in range(num_slices):
        for j in range(slice_length):
            # the ground truth labels in the neighborhood
            gt_flat = ground_truth[i-NR:i+NR+1, j-NR:j+NR+1].flatten()
            # if its neighborhood is all labeled as ink, use it
            if np.count_nonzero(gt_flat) == neigh_count:
                ink_inds.append(coord_to_ind(i,j))
            # else if it has enough points above the threshold, consider it on the fragment
            elif np.count_nonzero(np.where(volume[i][j] > THRESH, volume[i][j], 0)) > MIN:
                no_ink_inds.append(coord_to_ind(i,j))

    return np.array(ink_inds + no_ink_inds)



def make_feature_input():
    input_vects_filename = "feat_cache/input-vects-in{}-back{}-neigh{}".format(
            CUT_IN, CUT_BACK, NR)
    input_neighs_filename = "feat_cache/input-neighs-in{}-back{}-neigh{}".format(
            CUT_IN, CUT_BACK, NR)
    try:
        vect_list = np.load(input_vects_filename+".npy")
        neigh_inds = np.load(input_neighs_filename+".npy")
        return vect_list, neigh_inds

    except Exception:
        pass


    v_len = (CUT_IN + CUT_BACK)
    max_ind = len(vol_list)
    vect_list = np.zeros((len(vol_list), v_len), dtype=np.int)
    neigh_inds = np.zeros((len(vol_list), neigh_count), dtype=np.int)

    for i in range(len(vol_list)):
        (sl,v) = ind_to_coord(i)
        srf = (surf_pts[sl][v] - CUT_BACK)
        cut = (surf_pts[sl][v] + CUT_IN)
        vect = vol_list[i][srf:cut]

        tmp_inds = []
        for j in range(sl-NR, sl+NR+1):
            for k in range(v-NR, v+NR+1):
                if coord_to_ind(j,k) < max_ind:
                    tmp_inds.append(coord_to_ind(j,k))

        vect_list[i][:len(vect)] = vect
        print("{} values in vect {}".format(np.count_nonzero(vect_list[i]), i))
        neigh_inds[i][:len(tmp_inds)] = tmp_inds
    np.save(input_vects_filename, vect_list)
    np.save(input_neighs_filename, neigh_inds)
    return vect_list, neigh_inds




def make_train_pic(inds):
    train_pic = np.zeros((num_slices, slice_length), dtype=np.uint16)
    for ind in inds:
        train_pic[ind_to_coord(ind)] = 65535
    tiff.imsave("predictions/train.tif", train_pic)



def make_train_truth_pic(inds, truths):
    truth_pic = np.zeros((num_slices, slice_length), dtype=np.uint16)
    for i in range(len(inds)):
        truth = truths[i]
        truth_pic[ind_to_coord(inds[i])] = (truth+1)*(65534/2)
    tiff.imsave("predictions/train_truth.tif", truth_pic)



def make_full_pred_pic(preds, feat='all', strip=''):
    predicted_pic = np.zeros((num_slices, slice_length), dtype=np.uint16)
    pred_range = (min(preds), max(preds))
    print("pred range: {}".format(pred_range))
    for i in range(len(preds)):
        (sl,v) = ind_to_coord(i)
        pred = preds[i]
        predicted_pic[sl,v] = (pred *65535)
    pic_range = (np.min(predicted_pic), np.max(predicted_pic))
    print("pic range: {}".format(pic_range))
    tiff.imsave("predictions/prediction-in{}-back{}-neigh{}-feat{}{}.tif".format(
        CUT_IN, CUT_BACK, NR, feat, strip), predicted_pic)



def learn_lr(train_feats, train_gt, all_feats):
    clf = LogisticRegression()
    clf.fit(train_feats, train_gt)
    preds = clf.predict_proba(all_feats)[:,1]
    #preds = clf.predict(all_feats)
    return preds



def learn_svm(train_feats, train_gt, all_feats):
    clf = svm.SVR()
    clf.fit(train_feats, train_gt)
    preds = clf.predict(all_feats)
    return preds



def learn_tf(train_feats, train_gt, test_feats, test_gt, all_feats):
    # parameters
    learning_rate = 0.001
    training_epochs = 10
    batch_size = 100
    display_step = 1

    # network
    n_hidden_1 = 10
    n_hidden_2 = 256
    n_input = 10
    n_classes = 2

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
    }
    pred = multilayer_perceptron(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batches = int(len(train_feats) / batch_size)
            for i in range(1,total_batches):
                (batch_x, batch_y) = (train_feats[(i-1)*batch_size:i*batch_size],
                    train_gt[(i-1)*batch_size:i*batch_size])
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                avg_cost += c / total_batches
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.4f}".format(avg_cost))
        print("finished optimization")

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: test_feats, y: test_gt}))

        #feed_dict = {x : all_feats}
        #classifications = sess.run(y, feed_dict)
        #return classifications


def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(x, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer



def retrieve_volume():
    volume_filename = "volume"
    try:
        volume = np.load(volume_filename + ".npy")
        print("loaded volume from file")
    except Exception:
        print("could not load volume from file, extracting")
        volume = np.zeros((num_slices, slice_length, vol_depth), dtype=np.uint16)
        for sl in range(num_slices):
            slice_name = (data_path + "/vertical_rotated_slices/slice"
                        + "0000"[:4-len(str(sl))] + str(sl) + ".tif")
            current_slice = np.array(tiff.imread(slice_name))
            for v in range(slice_length):
                volume[sl][v] = current_slice[v]
    return volume



def retrieve_surf_pts(volume):
    surf_pts_filename = "surf-peaks-{}".format(THRESH)
    try:
        surf_pts = np.load(surf_pts_filename + ".npy")
        print("loaded surface points from file")
    except Exception:
        print("could not load surface points from file")
        surf_pts = np.zeros(output_dims, dtype=np.uint16)
        for sl in range(num_slices):
            for v in range(slice_length):
                surf_pts[sl][v] = extract_surf_peak(volume[sl][v])
    return surf_pts



def extract_surface(vect,length):
    to_return = np.zeros(length,dtype=np.uint16)
    thresh_vect = np.where(vect > THRESH, vect, 0)
    # find first peak above threshold
    surf_peak = argrelmax(thresh_vect)[0][0]
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



def coord_to_ind(sl,v):
    return (sl*slice_length) + v



if __name__ == "__main__":
    main()

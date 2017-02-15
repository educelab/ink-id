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
#import features
import tensorflow as tf
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
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
    global Cf
    THRESH = 20200 # the intensity threshold at which to extract the surface
    MIN = 2 # the minimum number of points above THRESH to be considered on the fragment
    CUT_IN = 8 # how far beyond the surface point to go
    CUT_BACK = 16 # how far behind the surface point to go
    STRP_RNGE = 64  # the width of the strip to train on
    NEIGH_RADIUS = 2  # the radius of the strip to train on
    Cf = 1e4 # the float value to use as the regularization parameter
    NR = NEIGH_RADIUS
    neigh_count = (2*NR+1)*(2*NR+1)

    global data_path
    global ground_truth
    data_path = "/home/jack/devel/ink-id/small-fragment-data"
    ground_truth = tiff.imread(data_path + "/registered/ground-truth-mask-full.tif")
    ground_truth = np.where(ground_truth == 255, 1, 0)

    global num_slices
    global slice_length
    global vol_depth
    global output_dims
    num_slices = ground_truth.shape[0]
    slice_length = ground_truth.shape[1]
    vol_depth = 286
    output_dims = (num_slices, slice_length)

    global volume
    global surf_pts
    volume = retrieve_volume()
    surf_pts = retrieve_surf_pts(volume)


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
    frag_truth = np.array([truth_list[i] for i in frag_inds])
    print("found {} points on fragment, {} ink".format(
        len(frag_inds), np.count_nonzero(frag_truth)))


    print("arranging input into neighborhoods")
    vect_list, neigh_inds = make_feature_input()
    # create a list of all samples on the fragment
    #frag_truth = [(1,0)]*len(ink_coords) + [(0,1)]*len(no_ink_coords)
    print("arranged input")


    print("extracting features")
    #all_feats = features.extract_for_list(vect_list, neigh_inds, CUT_IN, CUT_BACK, NR)
    all_feats = vect_list.astype(np.float32)


    print("splitting train/test sets")
    # make train and test
    n = len(frag_inds)
    k1 = int(.6*n)
    k2 = int(.1*n)
    np.random.shuffle(frag_inds)
    train_inds = (np.array(frag_inds[:k1]))
    #train_inds = np.sort(train_inds)
    test_inds = np.array(frag_inds[k1:(k1+k2)])
    #val_inds = inds[int(.8*n):]

    train_x = all_feats[train_inds]
    train_y = truth_list[train_inds]
    assert len(train_x) == len(train_y)
    test_x = all_feats[test_inds]
    test_y = truth_list[test_inds]
    assert len(test_x) == len(test_y)

    print("training models for predictions")
    #preds = learn_lr(train_x, train_y, all_feats)
    #make_full_pred_pic(preds, name="lr-C{}-".format(Cf))
    rec = 5
    learn_r = 0.001
    batch_s = 256
    preds = learn_tf_cnn(rec, learn_r, batch_s, train_x, train_y, test_x, test_y, all_feats)
    make_full_pred_pic(preds, name="tf-cnn-rec{}-batch{}-learn{}-".format(rec, batch_s, learn_r))
    #preds = learn_tf_mlp(train_x, train_y, all_feats)
    #make_full_pred_pic(preds, name="tf-mlp-")
    
    ''' 
    # use this loop for single-feature training
    for i in range(all_feats.shape[1]):
        tmp_x = train_x[:,i].reshape(-1,1)
        tmp_feats = all_feats[:,i].reshape(-1,1)
        #preds = tmp_feats
        preds = learn_tf_mlp(tmp_x, train_y, tmp_feats)
        make_full_pred_pic(preds, name="tf-mlp-", feat=str(i))

    # use this loop for strip training
    preds = np.zeros(len(all_feats), dtype=np.float32)
    strip_length = STRP_RNGE * slice_length
    for strip in range(0, int(num_slices / STRP_RNGE)):
        min_ind = strip * strip_length
        max_ind = min_ind + strip_length
        train_strip_inds = np.where(np.logical_and(
            train_inds>=min_ind-int(strip_length/2),
            train_inds<=max_ind+int(strip_length/2)))[0]

        tmp_x = [train_x[ind] for ind in train_strip_inds]
        tmp_y = [train_y[ind] for ind in train_strip_inds]
        tmp_feats = all_feats[min_ind:max_ind]

        num_inks = np.count_nonzero(tmp_y)
        num_no_inks = len(tmp_y) - num_inks
        if(num_inks > 20 and num_no_inks > 20):
            #preds[min_ind:max_ind] = learn_tf_tutor(tmp_x, tmp_y, test_x, test_y, tmp_feats)
            preds[min_ind:max_ind] = learn_lr(tmp_x, tmp_y, tmp_feats)
    make_full_pred_pic(preds, "tf-mlp-", strip='-strip{}'.format(STRP_RNGE))
    #make_full_pred_pic(preds, "lr-C{}-".format(Cf), strip='-strip{}'.format(STRP_RNGE))
    '''
    print("full script took {:.2f} seconds".format(time.time() - start_time))



def learn_lr(train_feats, train_gt, all_feats):
    clf = LogisticRegression(C=Cf)
    clf.fit(train_feats, train_gt)
    preds = clf.predict_proba(all_feats)[:,1]
    #preds = clf.predict(all_feats)
    return preds



def learn_svm(train_feats, train_gt, all_feats):
    clf = svm.SVR()
    clf.fit(train_feats, train_gt)
    preds = clf.predict(all_feats)
    return preds



def learn_tf_cnn(rec, learn_r, batch_s, train_feats, train_gt, test_feats, test_gt, all_feats):
    train_y = make_2d_gt(train_gt)
    # Params
    learning_rate = learn_r
    batch_size = batch_s 
    display_step = 100

    n_input = train_feats.shape[1]
    n_classes = 2

    x = tf.placeholder(tf.float32, [None, n_input])
    y_ = tf.placeholder(tf.float32, [None, n_classes])

    weights = {
        'wc1': tf.Variable(tf.random_normal([rec, 1, 1, 32])),
        'wc2': tf.Variable(tf.random_normal([rec, 1, 32, 64])),
        'wd1': tf.Variable(tf.random_normal([6*64, 256])),
        'out': tf.Variable(tf.random_normal([256, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([256])),
        'out': tf.Variable(tf.random_normal([n_classes])),
    }

    y = conv_net(x, weights, biases)
    print("x shape: {}".format(x.get_shape().as_list()))
    print("y_ shape: {}".format(y_.get_shape().as_list()))
    print("y shape: {}".format(y.get_shape().as_list()))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    steps = 0

    for i in range(1,int(len(train_feats) / batch_size)):
        batch_xs, batch_ys = (train_feats[(i-1)*batch_size: (i*batch_size)],
                                train_y[(i-1)*batch_size: (i*batch_size)])
        _, c = sess.run([optimizer,cost], feed_dict={x: batch_xs, y_: batch_ys})
        steps += 1
        if(steps % display_step == 0):
            print("cost is {} at step {}".format(c,i))


    test_preds = sess.run(y, feed_dict={x:test_feats})[:,1]
    conf_mat = confusion_matrix([0],[1])#test_preds, test_gt)
    print("test results: \n{}".format(conf_mat))
    print("test_preds: {}".format(test_preds[:10]))
    print("test_gt : {}".format(test_gt[:10]))

    predictions = np.zeros(all_feats.shape[0])
    for i in range(0, int(len(all_feats) / batch_size)-1):
        predictions[(i*batch_size):((i+1)*batch_size)] = sess.run(y, feed_dict={
            x:all_feats[(i*batch_size):((i+1)*batch_size)]})[:,1]
    

    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y, 1))
    p_min = np.min(predictions)
    p_max = np.max(predictions)
    return np.nan_to_num((predictions - p_min) / (p_max - p_min))


def conv2d(x, W, b, stride=1):
    x = tf.nn.conv2d(x, W, strides=[1, stride,stride, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')


def conv_net(x, weights, biases):
    x = tf.reshape(x, shape=[-1,(CUT_IN+CUT_BACK),1,1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1,k=2)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2,k=2)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out



def learn_tf_lr(train_feats, train_gt, test_feats, test_gt, all_feats):
    # parameters
    learning_rate = 0.01
    batch_size = 100

    # convert truth labels into truth tuples
    train_y = make_2d_gt(train_gt)
    #test_y = make_2d_gt(test_gt)

    repdim = all_feats.shape[1]
    x = tf.placeholder(tf.float32, [None, repdim])
    W = tf.Variable(tf.zeros([repdim, 2]))
    b = tf.Variable(tf.zeros([2]))
    # the model
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # the answewrs
    y_ = tf.placeholder(tf.float32, [None, 2])
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(1,int(len(train_feats) / batch_size)):
        batch_xs, batch_ys = (train_feats[(i-1)*batch_size: (i*batch_size)],
                                train_y[(i-1)*batch_size: (i*batch_size)])
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    #prediction = tf.argmax(y,1)
    prediction = y
    return sess.run(prediction, feed_dict={x:all_feats})[:,0]




def learn_tf_mlp(train_feats, train_gt, all_feats):
    # parameters
    learning_rate = 0.001
    batch_size = 64
    n_input = train_feats.shape[1]
    n_hidden_1 = train_feats.shape[1]
    n_hidden_2 = train_feats.shape[1] * 8
    n_classes = 2

    # Store layers weight & bias
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

    # convert truth labels into truth tuples
    train_y = make_2d_gt(train_gt)
    x = tf.placeholder(tf.float32, [None, n_input])

    # the answewrs
    y_ = tf.placeholder(tf.float32, [None, n_classes])

    # the model
    y = multilayer_perceptron(x, weights, biases)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(1,int(len(train_feats) / batch_size)):
        batch_xs, batch_ys = (train_feats[(i-1)*batch_size: (i*batch_size)],
                                train_y[(i-1)*batch_size: (i*batch_size)])
        sess.run([optimizer,cost], feed_dict={x: batch_xs, y_: batch_ys})

    predictions = sess.run(y, feed_dict={x:all_feats})[:,1]
    p_min = np.min(predictions)
    p_max = np.max(predictions)
    print("range before normalize: {} - {}".format(p_min, p_max))
    return np.nan_to_num((predictions - p_min) / (p_max - p_min))




def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(x, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer



def make_2d_gt(gt):
    gt_2d = np.zeros((gt.shape[0], 2), dtype=np.int)
    gt_2d[:,0] = gt
    gt_2d[gt==0, 1] = 1
    return gt_2d 



def retrieve_volume():
    volume_filename = "volume"
    try:
        volume = np.load(volume_filename + ".npy")
        print("loaded volume from file")
    except Exception:
        print("could not load volume from file, extracting")
        volume = np.zeros((num_slices, slice_length, vol_depth), dtype=np.uint16)
        for sl in range(num_slices):
            slice_name = (data_path + "/flatfielded-slices/slice"
                        + "0000"[:4-len(str(sl))] + str(sl) + ".tif")
            current_slice = np.array(tiff.imread(slice_name))
            for v in range(slice_length):
                volume[sl][v] = current_slice[v]
        np.save(volume_filename, volume)
    return volume



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



def make_full_pred_pic(preds, name='', feat='all', strip=''):
    print("preds: {}".format(preds.shape))
    predicted_pic = np.zeros((num_slices, slice_length), dtype=np.uint16)
    pred_range = (min(preds), max(preds))
    print("pred range: {}".format(pred_range))
    for i in range(len(preds)):
        (sl,v) = ind_to_coord(i)
        pred = preds[i]
        predicted_pic[sl,v] = (pred *65535)
    pic_range = (np.min(predicted_pic), np.max(predicted_pic))
    print("pic range: {}".format(pic_range))
    tiff.imsave("predictions/{}prediction-in{}-back{}-neigh{}-feat{}{}.tif".format(
        name, CUT_IN, CUT_BACK, NR, feat, strip), predicted_pic)



def retrieve_surf_pts(volume):
    surf_pts_filename = "surf-points-{}".format(THRESH)
    try:
        surf_pts = np.load(surf_pts_filename + ".npy")
        print("loaded surface points from file")
    except Exception:
        print("could not load surface points from file")
        surf_pts = np.zeros(output_dims, dtype=np.uint16)
        for sl in range(num_slices):
            for v in range(slice_length):
                try:
                    surf_pts[sl][v] = extract_surface(volume[sl][v])
                except Exception:
                    # no definitive peak/valley on surface
                    pass
        np.save(surf_pts_filename, surf_pts)
        print("extracted surface points and saved to file")
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

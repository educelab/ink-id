'''
trainvol.py
a data structure to interface between tensorflow training and raw volumetric data
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"

import numpy as np
import time
import tifffile as tiff
from skimage.transform import resize

class TrainVol(object):

    def __init__(self, volume_fn, truth_fn, CUT_IN, CUT_BACK, NR, THRESH, RSZ, DEPTH=0):
        print("initializing TrainVol...")
        self.current_spot = 0
        self.fragment_buffer = 4
        self.NR = NR
        self.C_IN = CUT_IN
        self.C_BACK = CUT_BACK
        self.RSZ = RSZ
        self.DEPTH = DEPTH

        print("loading raw volume...")
        self.raw_volume = np.load(volume_fn).astype(np.float64)
        x,y,z = self.raw_volume.shape
        rx = int(x/RSZ)
        ry = int(y/RSZ)
        rz = int(z/RSZ)
        print("resizing volume...")
        self.raw_volume = resize(self.raw_volume, (rx,ry,rz)).astype(np.uint16)
        print("loading ground truth...")
        self.raw_ground_truth = np.load(truth_fn).astype(np.float64)
        self.raw_ground_truth = resize(self.raw_ground_truth, (rx,ry)).astype(np.uint8)
        print("adjusting ground truth for neighborhood radius {}...".format(self.NR))
        self.raw_ground_truth = np.where(self.raw_ground_truth > 0, 1, 0)
        self.ground_truth = np.zeros(self.raw_ground_truth.shape, dtype=np.float32)
        for i in range(self.NR, self.ground_truth.shape[0] - self.NR):
            for j in range(self.NR, self.ground_truth.shape[1] - self.NR):
                # make sure all labels in the neighborhood are unanimously ink or not
                t_mean = np.mean(self.raw_ground_truth[i-NR:i+NR, j-NR:j+NR])
                if t_mean > .9:
                    self.ground_truth[i,j] = .99
                elif t_mean < .1:
                    self.ground_truth[i,j] = .00
                '''
                else:
                    self.ground_truth[i,j] = t_mean / 2
                '''
        self.flt_volume = self.raw_volume.astype(np.float32)
        self.train_style = 'random'
        np.random.seed(int(time.time())) # use a different seed each time
        self.total_vects = self.raw_volume.shape[0]*self.raw_volume.shape[1]
        print("finding fragment surface at threshold {}...".format(THRESH))
        self.find_frag_surf_at_thresh(THRESH)
        print("DONE initializing TrainVol")



    def next_batch(self, batch_size):
        """Returns 'batch_size' 2D training samples"""
        width = 2*self.NR
        height = self.C_IN + self.C_BACK
        to_return_x = np.zeros((batch_size, width, height), dtype=np.float32)
        to_return_y = np.zeros((batch_size, 2), dtype=np.float32)

        # if everything has been trained on, loop back
        if(self.current_spot + batch_size > self.total_vects):
            self.current_spot = 0
            np.random.shuffle(self.train_inds)

        start = self.current_spot
        end = start + batch_size 

        batch_inds = self.train_inds[start:end]
        c = 0
        NR = self.NR
        IN = self.C_IN
        BK = self.C_BACK
        for ind in batch_inds:
            row,col = self.ind_to_coord(ind)
            surf = self.surf_pts[row, col]
            to_put = self.flt_volume[row, col-NR:col+NR, surf-BK:surf+IN]
            if to_put.shape == to_return_x[c].shape:
                if (np.random.randint(10) % 2 == 0):
                    to_put = np.flipud(to_put)

                to_return_x[c] = to_put
                to_return_y[c][0] = 1 - self.ground_truth[row, col] # non-ink
                to_return_y[c][1] = self.ground_truth[row, col] # ink

            c += 1

        self.current_spot = end
        return (to_return_x, to_return_y)



    def next_batch_3d(self, batch_size):
        """Returns 'batch_size' 3D training samples"""
        #Use self.DEPTH to create 3D data. Also, use data from reverse side of fragment

        width = 2*self.NR
        height = self.C_IN + self.C_BACK
        depth = self.DEPTH * 2
        to_return_x = np.zeros((batch_size, width, height, depth), dtype=np.float32)
        to_return_y = np.zeros((batch_size, 2), dtype=np.float32)

        # if everything has been trained on, loop back
        if(self.current_spot + batch_size > self.total_vects):
            self.current_spot = 0
            np.random.shuffle(self.train_inds)

        start = self.current_spot
        end = start + batch_size / 2 # using the reverse side halves the required indeces

        batch_inds = self.train_inds[start:end]
        c = 0
        NR = self.NR
        IN = self.C_IN
        BK = self.C_BACK
        DP = self.DEPTH
        for ind in batch_inds:
            row,col = self.ind_to_coord(ind)
            surf = self.surf_pts[row, col]
            to_put = self.flt_volume[row-DP:row+DP, col-NR:col+NR, surf-BK:surf+IN]
            if to_put.shape == to_return_x[c].shape:
                to_return_x[c] = to_put
                to_return_y[c][0] = 1 - self.ground_truth[row, col] # non-ink
                to_return_y[c][1] = self.ground_truth[row, col] # ink

            # add the reverse side
            rev_surf = self.surf_pts_reverse[row, col]
            to_put_rev = self.flt_volume[row-DP:row+DP, col-NR:col+NR, rev_surf-IN:rev_surf+BK]
            np.flip(to_put_rev, axis=2)
            if to_put_rev.shape == to_return_x[c+1].shape:
                to_return_x[c+1] = to_put
                to_return_y[c+1][0] = 1.00 # all points on reverse side are 100% non-ink
                to_return_y[c+1][1] = .00 # and 0% ink

            c += 2

        self.current_spot = end
        return (to_return_x, to_return_y)




    def get_predict_batch(self):
        width = 2*self.NR
        height = self.C_IN + self.C_BACK
        to_return_x = np.zeros((self.total_vects, width, height), dtype=np.float32)
        NR = self.NR
        IN = self.C_IN
        BK = self.C_BACK

        for ind in range(self.total_vects):
            row,col = self.ind_to_coord(ind)
            surf = self.surf_pts[row, col]
            to_put = self.flt_volume[row, col-NR:col+NR, surf-BK:surf+IN]
            if to_put.shape == to_return_x[ind].shape:
                to_return_x[ind] = to_put


        return to_return_x




    def get_output_shape(self):
        return (self.flt_volume.shape[0], self.flt_volume.shape[1])




    def make_train_with_style(self, style, prob=0.6):
        ind_pool = np.zeros(self.total_vects, dtype=np.int)

        print("creating fragment mask")
        for row in range(self.raw_volume.shape[0]):
            for col in range(self.raw_volume.shape[1]):
                if self.frag_mask[row,col]:
                    ind = self.coord_to_ind(row,col)
                    ind_pool[ind] = ind
        ind_pool = ind_pool[np.nonzero(ind_pool)[0]]
        print("created fragment mask")

        if style == 'random':
            # train on a random sample of dropout% points
            self.train_style = 'random'
            n = ind_pool.shape[0]
            np.random.shuffle(ind_pool)
            train_inds = ind_pool[:int(prob*n)]
            np.random.shuffle(train_inds)

        elif style == 'rhalf':
            # only train on the right half
            self.train_style = 'rhalf'
            split = self.raw_volume.shape[1] / 2
            train_inds = ind_pool[np.where(ind_pool % 2 > split)]
            np.random.shuffle(train_inds)

        elif style == 'lhalf':
            # only train on the left half
            self.train_style = 'lhalf'
            split = self.raw_volume.shape[1] / 2
            train_inds = ind_pool[np.where(ind_pool % 2 < split)]
            np.random.shuffle(train_inds)

        self.train_inds = train_inds
        train_coords = [self.ind_to_coord(i) for i in train_inds]
        truth_pic = np.zeros(self.ground_truth.shape, dtype=np.uint16)
        for (i,j) in train_coords:
            truth_pic[i,j] = (self.ground_truth[i,j] + 1) * (65534 / 2)
        tiff.imsave('predictions/latest-training-truth.tif', truth_pic)



    def find_frag_surf_at_thresh(self, thresh):
        maxes = np.max(self.raw_volume, axis=2)
        self.frag_mask = np.zeros(maxes.shape, dtype=np.uint16)
        self.surf_pts = np.zeros(maxes.shape, dtype=np.int)
        self.surf_pts_reverse = np.zeros(maxes.shape, dtype=np.int)

        n = self.fragment_buffer  # neighborhood to verify

        for i in range(n, self.raw_volume.shape[0]-n-1):
            for j in range(n, self.raw_volume.shape[1]-n-1):
                # if points in the point's neighborhood exceed the threshold
                if np.min(maxes[i-n:i+n+1, j-n:j+n+1]) > thresh:
                    # mark it as a point on the fragment
                    self.frag_mask[i,j] = 1
                    surface_vect = np.where(self.raw_volume[i,j] > thresh)[0]
                    # store the surface's "start point"
                    self.surf_pts[i,j] = surface_vect[0]
                    # and the surface's "end point"
                    self.surf_pts_reverse[i,j] = surface_vect[-1]



    def ind_to_coord(self, ind):
        width = self.raw_volume.shape[1]
        col = ind % width
        row = int(ind / width)
        return row,col



    def coord_to_ind(self, row, col):
        width = self.raw_volume.shape[1]
        return (row*width)+col



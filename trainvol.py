'''
trainvol.py
a data structure to interface between tensorflow training and raw volumetric data
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"

import numpy as np

class TrainVol(object):

    def __init__(self, volume_fn, truth_fn):
        print("initializing TrainVol")
        self.current_spot = 0
        self.fragment_buffer = 4
        self.NR = 2
        self.C_IN = 4
        self.C_BACK = 4

        self.raw_volume = np.load(volume_fn)
        self.ground_truth = np.load(truth_fn)
        self.flt_volume = self.raw_volume.astype(np.float32)
        self.train_style = 'drop'
        self.total_vects = self.raw_volume.shape[0]*self.raw_volume.shape[1]
        self.frag_mask, self.surf_pts = self.find_frag_surf_at_thresh(20500)
        print("done initializing TrainVol")



    def next_batch(self, batch_size):
        width = 2*self.NR + 1
        height = self.C_IN + self.C_BACK
        to_return_x = np.zeros((batch_size, width, height), dtype=np.float32)
        to_return_y = np.zeros((batch_size, 2), dtype=np.float32)

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
            print("row, col, surf = {}, {}, {}".format(row, col, surf))
            to_put = self.flt_volume[row, col-NR:col+NR+1, surf-BK:surf+IN]
            to_return_x[c] = to_put
            to_return_y[c][0] = self.ground_truth[row, col]
            to_return_y[c][1] = 1 - self.ground_truth[row, col]
            c += 1

        self.current_spot = end
        return (to_return_x, to_return_y)




    def make_train_with_style(self, style, dropout=0.6):
        ind_pool = np.zeros(self.total_vects, dtype=np.int)

        print("creating fragment mask")
        for row in range(self.raw_volume.shape[0]):
            for col in range(self.raw_volume.shape[1]):
                if self.frag_mask[row,col]:
                    ind = self.coord_to_ind(row,col)
                    ind_pool[ind] = ind
        ind_pool = ind_pool[np.nonzero(ind_pool)[0]]
        print("created fragment mask")

        if style == 'drop':
            # train on a random sample of dropout% points
            self.train_style = 'drop'
            n = ind_pool.shape[0]
            np.random.shuffle(ind_pool)
            train_inds = ind_pool[:(dropout*n)]

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



    def find_frag_surf_at_thresh(self, thresh):
        maxes = np.max(self.raw_volume, axis=2)
        frag_mask = np.zeros(maxes.shape, dtype=np.uint16)
        surf_pts = np.zeros(maxes.shape, dtype=np.int)

        n = self.fragment_buffer  # neighborhood to verify

        for i in range(n, self.raw_volume.shape[0]-n-1):
            for j in range(n, self.raw_volume.shape[1]-n-1):
                if np.min(maxes[i-n:i+n+1, j-n:j+n+1].flatten()) > thresh:
                    frag_mask[i,j] = 1
                    vector = self.raw_volume[i,j]
                    surf_pts[i,j] = np.where(vector > thresh)[0][0]

        return frag_mask, surf_pts



    def ind_to_coord(self, ind):
        width = self.raw_volume.shape[1]
        col = ind % width
        row = int(ind / width)
        return row,col



    def coord_to_ind(self, row, col):
        width = self.raw_volume.shape[1]
        return (row*width)+col



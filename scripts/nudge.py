'''
nudge.py
create a nudged version of the volume, increasing values at ink points
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"

import numpy as np
import tifffile as tiff
import os
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
from scipy.stats import norm
import imageio


def main():
    slices_dir = '/home/jack/devel/volcart/simulated-accuracy/blank-papyrus-layers/layers-top-down/'

    slices_list = os.listdir(slices_dir)
    slices_list.sort()
    volume = []
    print("Constructing volume...")
    for s in slices_list:
        volume.append(np.array(imageio.imread(slices_dir+s)))
    vol = np.array(volume)

    output_dir = '/home/jack/devel/volcart/simulated-accuracy/nudge-'
    output = np.zeros(vol.shape, dtype=np.uint16)

    # parameters
    loc = 0
    scale = 4 # how much to stretch the curve, lower = taller curve, higher = shorter/wider
    increase_percentages = np.array([40, 30, 20])
    increase_decimals = increase_percentages / 100
    neigh = 2
    thresh = 1
    reach_in = 2
    reach_back = 2
    span = max(reach_in, reach_back)
    show_demo = False
    shown_demo = False 

    for p in preds:
        name = p[-40:]
        print("Nudging with prediction {}".format(name))
        truth_mask = imageio.imread(p)
        #truth_mask = imageio.imread('/home/jack/devel/volcart/simulated-accuracy/layer-mask-gt.png')
        print("Creating nudge distribution...")
        # create the distribution array
        distribute = [0.0] * (span+1)
        before = np.zeros(truth_mask.shape, dtype=np.uint16)
        after = np.zeros(truth_mask.shape, dtype=np.uint16)
        cap = np.iinfo(vol.dtype).max
        vol_min = np.min(np.where(vol > 0, vol, cap))
        vol_max = np.max(vol)
        vol_range = (vol_max - vol_min)
        truth_value = np.max(truth_mask)

        for i in range(len(distribute)):
            # the initial distribution
            distribute[i] = norm.pdf(i, loc, scale)

        for increase in increase_decimals:
            print("Performing nudge of {}...".format(increase))
            # re-initialize everything
            vol = np.array(volume)
            outvol = np.copy(vol)
            before = np.zeros(truth_mask.shape[0:2], dtype=np.uint16)
            after = np.zeros(truth_mask.shape[0:2], dtype=np.uint16)

            target_increase = increase * vol_range
            increase_parameter = (target_increase / distribute[0])
            out_ceiling = np.iinfo(np.uint16).max
            # for example if the target increase is 1.0% (.010),
            # target_increase = 65535*.01 = 655.35
            # increase_parameter = 655.35 / .19 = 3285


            # main loop
            for i in range(neigh, vol.shape[0] - neigh):
                for j in range(neigh, vol.shape[1] - neigh):
                    vector = vol[i,j]
                    truth_weight = np.mean(truth_mask[i-neigh:i+neigh, j-neigh:j+neigh]) / truth_value

                    # set everything below threshold to 0
                    thresh_vect = np.where(vector > thresh, vector, 0)
                    try:
                        #peak = argrelmax(thresh_vect)[0][0]
                        peak = 10
                        before[i,j] = vector[peak]

                        # nudge each point around the peak
                        for x in range(peak - reach_back, peak):
                            diff = abs(peak - x)
                            proportion = float(diff) / float(reach_back)
                            aligned_index = int(proportion * span)
                            dist_weight = distribute[aligned_index]
                            increase_amt = int(increase_parameter * truth_weight * dist_weight)
                            vector[x] = min(vector[x]+increase_amt, out_ceiling)
                        for x in range(peak, peak + reach_in):
                            diff = abs(peak - x)
                            dist_weight = distribute[diff]
                            increase_amt = int(increase_parameter * truth_weight * dist_weight)
                            vector[x] = min(vector[x]+increase_amt, out_ceiling)


                        outvol[i,j] = vector
                        after[i,j] = vector[peak]
                        if show_demo and not shown_demo and truth_weight > .9:
                            xs = np.arange(vol.shape[2])
                            plt.plot(thresh_vect, color='b')
                            plt.plot(vector, color='g')
                            plt.show()
                            shown_demo = True

                    except IndexError:
                        # for when no argrelmax exists
                        pass

                #progress update
                if (i % int((vol.shape[0] - neigh) / 10) == 0):
                    print("finished rows 0 to {} out of {} for increase {}".format(
                        i, vol.shape[0] - neigh, increase))

            # output
            current_output_dir = (output_dir + "{:.2f}%-{}".format(increase * 100, name) + "/")
            try:
                os.mkdir(current_output_dir)
            except Exception:
                pass

            # 1: save the volume and surface images
            '''np.save(current_output_dir+"volume-nudged-{:.2f}%".format(
                increase*100), outvol)'''
            imageio.imwrite(current_output_dir+"values-before-nudge-{:.2f}%.tif".format(
                increase*100), before)
            imageio.imwrite(current_output_dir+"values-after-nudged-{:.2f}%.tif".format(
                increase*100), after)

            # 2: save the slices
            '''
            slice_dir = current_output_dir + "/slices/"
            try:
                os.mkdir(slice_dir)
            except Exception:
                pass
            
            for sl in range(outvol.shape[0]):
                zeros = len(str(sl))
                tiff.imsave(slice_dir+"slice" + "0000"[:4-zeros] + str(sl), outvol[sl])
            '''
            # 3: save the planet
            #TODO



preds = [
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp10-fp203/prediction-pr0.892-rec1.000-acc0.989.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp126-fp192/prediction-pr0.634-rec0.500-acc0.926.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp72-fp197/prediction-pr0.824-rec0.933-acc0.975.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp66-fp197/prediction-pr0.827-rec0.953-acc0.977.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp122-fp192/prediction-pr0.655-rec0.546-acc0.931.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp0-fp204/prediction-pr0.900-rec1.000-acc0.990.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp26-fp201/prediction-pr0.874-rec0.999-acc0.986.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp30-fp201/prediction-pr0.874-rec0.998-acc0.986.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp70-fp197/prediction-pr0.825-rec0.940-acc0.976.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp2-fp203/prediction-pr0.892-rec1.000-acc0.989.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp94-fp195/prediction-pr0.778-rec0.812-acc0.961.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp114-fp193/prediction-pr0.702-rec0.633-acc0.941.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp12-fp202/prediction-pr0.883-rec1.000-acc0.988.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp98-fp194/prediction-pr0.758-rec0.782-acc0.956.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp92-fp195/prediction-pr0.781-rec0.827-acc0.962.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp4-fp203/prediction-pr0.892-rec1.000-acc0.989.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp40-fp200/prediction-pr0.864-rec0.994-acc0.985.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp106-fp194/prediction-pr0.740-rec0.712-acc0.950.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp78-fp196/prediction-pr0.809-rec0.908-acc0.971.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp20-fp202/prediction-pr0.883-rec0.999-acc0.988.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp22-fp201/prediction-pr0.874-rec0.999-acc0.986.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp96-fp195/prediction-pr0.775-rec0.797-acc0.959.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp68-fp197/prediction-pr0.826-rec0.947-acc0.976.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp38-fp200/prediction-pr0.864-rec0.995-acc0.985.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp36-fp200/prediction-pr0.864-rec0.996-acc0.985.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp42-fp200/prediction-pr0.864-rec0.993-acc0.985.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp80-fp196/prediction-pr0.807-rec0.898-acc0.970.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp64-fp198/prediction-pr0.839-rec0.959-acc0.979.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp34-fp200/prediction-pr0.864-rec0.997-acc0.985.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp76-fp196/prediction-pr0.810-rec0.917-acc0.972.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp16-fp202/prediction-pr0.883-rec1.000-acc0.988.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp48-fp199/prediction-pr0.853-rec0.988-acc0.983.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp104-fp194/prediction-pr0.745-rec0.731-acc0.952.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp102-fp194/prediction-pr0.750-rec0.748-acc0.953.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp62-fp198/prediction-pr0.840-rec0.964-acc0.979.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp88-fp195/prediction-pr0.787-rec0.853-acc0.965.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp86-fp195/prediction-pr0.789-rec0.865-acc0.966.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp58-fp198/prediction-pr0.841-rec0.973-acc0.980.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp90-fp195/prediction-pr0.784-rec0.840-acc0.963.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp108-fp193/prediction-pr0.721-rec0.693-acc0.946.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp74-fp197/prediction-pr0.823-rec0.925-acc0.974.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp110-fp193/prediction-pr0.715-rec0.673-acc0.944.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp52-fp199/prediction-pr0.853-rec0.983-acc0.983.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp18-fp202/prediction-pr0.883-rec1.000-acc0.988.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp46-fp199/prediction-pr0.854-rec0.990-acc0.983.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp24-fp201/prediction-pr0.874-rec0.999-acc0.986.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp124-fp192/prediction-pr0.645-rec0.523-acc0.929.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp50-fp199/prediction-pr0.853-rec0.986-acc0.983.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp60-fp198/prediction-pr0.840-rec0.969-acc0.980.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp118-fp192/prediction-pr0.672-rec0.591-acc0.935.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp8-fp203/prediction-pr0.892-rec1.000-acc0.989.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp32-fp201/prediction-pr0.874-rec0.997-acc0.986.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp82-fp196/prediction-pr0.805-rec0.888-acc0.969.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp112-fp193/prediction-pr0.709-rec0.653-acc0.943.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp28-fp201/prediction-pr0.874-rec0.998-acc0.986.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp120-fp192/prediction-pr0.664-rec0.569-acc0.933.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp54-fp198/prediction-pr0.842-rec0.980-acc0.981.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp100-fp194/prediction-pr0.754-rec0.765-acc0.955.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp44-fp199/prediction-pr0.854-rec0.992-acc0.983.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp84-fp196/prediction-pr0.803-rec0.877-acc0.968.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp6-fp203/prediction-pr0.892-rec1.000-acc0.989.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp14-fp202/prediction-pr0.883-rec1.000-acc0.988.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp56-fp198/prediction-pr0.841-rec0.977-acc0.981.png",
"/home/jack/devel/volcart/simulated-accuracy/all-simulated-accuracies/out-tp116-fp193/prediction-pr0.695-rec0.613-acc0.939.png",
]

if __name__ == "__main__":
    main()



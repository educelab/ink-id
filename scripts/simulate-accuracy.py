'''
simulate-accuracy.py
'''

import os
import numpy as np
from imageio import imread, imsave
from PIL.ImageOps import invert
from PIL import Image

# Initializations
surf_image_path = '/home/jack/devel/volcart/simulated-accuracy/blank-papyrus-layers/layers_mask.png'
surf_image = imread(surf_image_path)
surf_image = np.array(surf_image / np.max(surf_image), dtype=np.uint16) # set everything to 1 or 0
gt_image_path = '/home/jack/devel/volcart/simulated-accuracy/layer-mask-gt.png'
gt_image = imread(gt_image_path)[:,:,0]
invert_gt_image = np.array(invert(Image.fromarray(gt_image)))
pn_image_path = '/home/jack/devel/volcart/simulated-accuracy/perlin-noise-2008by5096-8octaves-0.5persistence.png'
pn_image = imread(pn_image_path)[:,:,0] * surf_image
out_image_path = '/home/jack/devel/volcart/simulated-accuracy'

accuracies_to_simulate = [.5, .6, .7, .8, .9, .95]
# (ink, blank) i.e. (true positive cutoff, false positive cutoff)
thresholds = np.zeros((64,2))
thresholds[:,0] = np.arange(0,128,2)
thresholds[:,1] = (( (1. - (thresholds[:,0] / 128.)) * 12.) + 192.)
thresholds = thresholds.astype(np.int)
            

total_points = pn_image.shape[0]*pn_image.shape[1]

# Main loop
for tp_thresh, fp_thresh in thresholds:
    ink_noise_mask = np.where(pn_image > tp_thresh, 1, 0)
    ink_mask = np.where(gt_image > 128, 1, 0)
    ink_above_noise = ink_mask * ink_noise_mask

    blank_noise_mask = np.where(pn_image > fp_thresh, 1, 0)
    blank_mask = np.where(invert_gt_image > 128, 1, 0)
    blank_above_noise = blank_mask * blank_noise_mask

    all_above_noise = blank_above_noise + ink_above_noise

    # Threshold for ink
    total_gt_ink_points = np.count_nonzero(ink_mask)
    total_simulated_ink_points = np.count_nonzero(ink_above_noise)
    ink_accuracy = total_simulated_ink_points / total_gt_ink_points
    
    # Threshold for papyrus/blank
    total_gt_blank_points = np.count_nonzero(blank_mask)
    total_simulated_fp = np.count_nonzero(blank_above_noise)
    fp_percentage = total_simulated_fp / total_gt_blank_points

    # Accuracy/Precision/Recall calculation
    correct_points = np.count_nonzero(np.where(all_above_noise == ink_mask, 1, 0))
    accuracy = correct_points / total_points
    precision = np.count_nonzero(ink_above_noise) / np.count_nonzero(all_above_noise)
    recall = np.count_nonzero(ink_above_noise) / np.count_nonzero(ink_mask)

    # Output
    print("Ink activations: {} / {} = {:.3f}".format(
        total_simulated_ink_points, total_gt_ink_points, ink_accuracy))
    print("False Positive activations: {} / {} = {:.3f}".format(
        total_simulated_fp, total_gt_blank_points, fp_percentage))
    print("Precision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("Accuracy: {:.3f}".format(accuracy))
    full_path_ink = '{}/out-tp{}-fp{}/ink-tp{:.3f}.png'.format(out_image_path, tp_thresh, fp_thresh, ink_accuracy)
    full_path_blank = '{}/out-tp{}-fp{}/blank-fp{:.3f}.png'.format(out_image_path, tp_thresh, fp_thresh, fp_percentage)
    full_path_all = '{}/out-tp{}-fp{}/prediction-pr{:.3f}-rec{:.3f}-acc{:.3f}.png'.format(
            out_image_path, tp_thresh, fp_thresh, precision, recall, accuracy)

    try:
        os.mkdir('{}/out-tp{}-fp{}'.format(out_image_path, tp_thresh, fp_thresh))
    except:
        pass
    imsave(full_path_ink, (ink_above_noise))
    imsave(full_path_blank, (blank_above_noise))
    imsave(full_path_all, (all_above_noise))

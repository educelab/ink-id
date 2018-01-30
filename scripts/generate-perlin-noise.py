'''
generate-perlin-noise.py
'''


import noise
import numpy as np
from scipy.misc import toimage
from scipy.misc import imread
from imageio import imwrite

gt_image_path = '/home/jack/devel/volcart/simulated-accuracy/layer-mask-gt.png'
gt = imread(gt_image_path)

# parameters for noise
shape = gt.shape
scale = 100.0
octaves = 8
persistence = 0.5
lacunarity = 2.0
out_image_path = '/home/jack/devel/volcart/simulated-accuracy/perlin-noise-{}by{}-{}octaves-{}persistence.png'.format(shape[0], shape[1], octaves, persistence)

noise_pic = np.zeros(shape)
# basic noise creation
for i in range(shape[0]):
    print("Making noise at row {}/{}".format(i, shape[0]))
    for j in range(shape[1]):
        noise_pic[i][j] = noise.pnoise2(i/scale,
                                        j/scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=shape[0],
                                        repeaty=shape[1],
                                        base=0)


imwrite(out_image_path, noise_pic)

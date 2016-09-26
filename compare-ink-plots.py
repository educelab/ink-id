'''
compare-ink-plots.py
Randomly walk through the fragment image, find points near the
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"


import tifffile as tiff
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from scipy.signal import argrelmax

NEIGH_SZ = 20 

data_path = "/Users/jack/dev/ink-id/small_fragment_data"
ref_photo = tiff.imread(data_path+"/registered/aligned-photo-contrast.tif")
ground_truth = tiff.imread(data_path+"/registered/ground-truth-mask.tif")
the_slice = tiff.imread(data_path+"/vertical_rotated_slices/slice0000.tif")
randy = 0

if len(sys.argv)== 1:
  #pick a random slice
  #make sure the slice has enough ground-truth, 10% of the slice
  while(np.count_nonzero(ground_truth[randy]) < (len(ground_truth[randy]) / 10)):
    randy = random.randint(0,len(ground_truth)-1)
    digits = len(str(randy))
    the_slice = tiff.imread(data_path+"/vertical_rotated_slices/slice" \
		    +"0000"[:4-digits]+str(randy)+".tif")
    print "Randomly chose slice " + str(randy)

elif len(sys.argv)==2:
  try:
    randy = int(sys.argv[1])
    digits = len(str(randy))
    the_slice_name = (data_path+"/vertical_rotated_slices/slice" \
		    +"0000"[:4-digits]+str(randy)+".tif")
    print(the_slice_name)
    the_slice = tiff.imread(the_slice_name)
  except Exception:
    sys.exit("Error retrieving slice")

else:
  sys.exit("Call me Bill O'Reilly, because I can't understand your arguments")




#rotate the slice to align visually
the_slice = np.rot90(the_slice)
#find the edge of the ink
pat = [0,255]
edge = [x for x in range(len(ground_truth[randy])-len(pat)) \
		if np.array_equal(ground_truth[randy][x:x+len(pat)], pat)]
print("edge index: " + str(x))


#grab an "inky" point close to the edge
ink_pt_x = (edge[0]+NEIGH_SZ)
ink_pt_y = (randy)
ink_pt = (ink_pt_x,ink_pt_y)
print("ink point: " + str(ink_pt))


#grab a "non-inky" point close to the edge
no_ink_pt_x = (edge[0]-NEIGH_SZ)
no_ink_pt_y = (randy)
no_ink_pt = (no_ink_pt_x,no_ink_pt_y)
print("no ink point: " + str(no_ink_pt))


#set threshold
THRESH = 20000
ink_vect = the_slice[:,ink_pt_x]
no_ink_vect = the_slice[:,no_ink_pt_x]

#anything in the vector below threshold gets set to 0
ink_vect = np.where(ink_vect > THRESH, ink_vect, 0)
no_ink_vect = np.where(no_ink_vect > THRESH, no_ink_vect, 0)

#find first and last peak in the new vector
#use scipy's fast argrelmax instead of writing my own
ink_vect_peaks = argrelmax(ink_vect)[0]
no_ink_vect_peaks = argrelmax(no_ink_vect)[0]

ink_vect_start = ink_vect_peaks[0]
ink_vect_end = ink_vect_peaks[-1]
no_ink_vect_start = no_ink_vect_peaks[0]
no_ink_vect_end = no_ink_vect_peaks[-1]


#figure 1,subplot 1: reference photo, ink/non-ink are dots
plt.figure(1)
plt.subplot(121)
plt.imshow(ref_photo,cmap="Greys_r")
#plt.plot([ink_pt_x,no_ink_pt_x],[ink_pt_y,no_ink_pt_y],color='g',marker='s')
#plt.plot((ink_pt_x),(ink_pt_y),'gs')
plt.plot((ink_pt_x),(ink_pt_y),color='g',marker='s')
plt.plot((no_ink_pt_x),(no_ink_pt_y),color='r',marker='s')

#figure 1,subplot 2: slice photo, ink/non-ink are lines
plt.subplot(122)
plt.imshow(the_slice,cmap="Greys_r")
height = len(the_slice[:,0])
plt.plot([ink_pt_x,ink_pt_x],[0,height],color='g',linewidth=2)
plt.plot([no_ink_pt_x,no_ink_pt_x],[0,height],color='r',linewidth=2)
plt.plot((ink_pt_x),(ink_vect_start),color='g',marker='s')
plt.plot((ink_pt_x),(ink_vect_end),color='g',marker='s')
plt.plot((no_ink_pt_x),(no_ink_vect_start),color='r',marker='s')
plt.plot((no_ink_pt_x),(no_ink_vect_end),color='r',marker='s')


#figure 2, subplot 1: ink plot
ymax = max([max(the_slice[:,ink_pt_x]), max(the_slice[:,no_ink_pt_x])])

plt.figure(2)
plt.subplot(211)
plt.plot(the_slice[:,ink_pt_x],color='g')

#figure 2, subplot 2: no-ink plot
plt.subplot(212)
plt.plot(the_slice[:,no_ink_pt_x],color='r')


#show dat plot
plt.show()

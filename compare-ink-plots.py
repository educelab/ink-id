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



'''handle a click in the plot'''
def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % \
          (event.button, event.x, event.y, event.xdata, event.ydata))
    selection = int(event.ydata)
    digits = len(str(selection))
    the_slice_name = (data_path+"/vertical_rotated_slices/slice" \
                +"0000"[:4-digits]+str(selection)+".tif")
                
    pat = [0,255]
    edge = [x for x in range(len(ground_truth[selection])-len(pat)) \
		if np.array_equal(ground_truth[selection][x:x+len(pat)], pat)]
    print("edge index: " + str(x))


    #grab an "inky" point close to the edge
    ink_pt_x = (edge[0]+NEIGH_SZ)
    ink_pt_y = (selection)
    ink_pt = (ink_pt_x,ink_pt_y)
    print("ink point: " + str(ink_pt))

    #grab a "non-inky" point close to the edge
    no_ink_pt_x = (edge[0]-NEIGH_SZ)
    no_ink_pt_y = (selection)
    no_ink_pt = (no_ink_pt_x,no_ink_pt_y)
    print("no ink point: " + str(no_ink_pt))
    
    plt.close()
    plot_slice_pts(the_slice_name,ink_pt_x, ink_pt_y, no_ink_pt_x, no_ink_pt_y)


def main():    
    if len(sys.argv)== 1:
      randy = 0
      #pick a random slice
      #make sure the slice has enough ground-truth, 10% of the slice
      while(np.count_nonzero(ground_truth[randy]) < (len(ground_truth[randy]) / 10)):
        randy = random.randint(0,len(ground_truth)-1)
        digits = len(str(randy))
        the_slice_name = data_path+"/vertical_rotated_slices/slice" + "0000"[:4-digits] + str(randy) + ".tif"
        print "Randomly chose " + the_slice_name

    elif len(sys.argv)==2:
      try:
        randy = int(sys.argv[1])
        digits = len(str(randy))
        the_slice_name = (data_path+"/vertical_rotated_slices/slice" \
                +"0000"[:4-digits]+str(randy)+".tif")
        print("Selected " + the_slice_name)
      except Exception:
        sys.exit("Error retrieving slice")

    else:
      sys.exit("Call me Bill O'Reilly, because I can't understand your arguments")
      
    #rotate the slice to align visually
    #the_slice = np.rot90(the_slice)
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
    
    plot_slice_pts(the_slice_name,ink_pt_x, ink_pt_y, no_ink_pt_x, no_ink_pt_y)



def plot_slice_pts(slice_name, ink_pt_x, ink_pt_y, no_ink_pt_x, no_ink_pt_y):

    the_slice = tiff.imread(slice_name)

    #set threshold
    THRESH = 20000
    ink_vect = the_slice[ink_pt_x]
    no_ink_vect = the_slice[no_ink_pt_x]

    #anything in the vector below threshold gets set to 0
    ink_vect = np.where(ink_vect > THRESH, ink_vect, 0)
    no_ink_vect = np.where(no_ink_vect > THRESH, no_ink_vect, 0)

    #find first and last peak in the new vector
    #use scipy's fast argrelmax instead of writing my own
    ink_vect_peaks = argrelmax(ink_vect)[0]
    no_ink_vect_peaks = argrelmax(no_ink_vect)[0]





    #figure 1,subplot 1: reference photo, ink/non-ink are dots
    fig1 = plt.figure(1)
    fig1.clf()
    p1 = fig1.add_subplot(111)
    p1.imshow(ref_photo,cmap="Greys_r")
    #plt.plot([ink_pt_x,no_ink_pt_x],[ink_pt_y,no_ink_pt_y],color='g',marker='s')
    #plt.plot((ink_pt_x),(ink_pt_y),'gs')
    p1.plot((ink_pt_x),(ink_pt_y),color='g',marker='s')
    p1.plot((no_ink_pt_x),(no_ink_pt_y),color='r',marker='s')
    cid = fig1.canvas.mpl_connect('button_press_event', onclick)

    #figure 2: slice image, ink and no-ink plots
    fig2 = plt.figure(2)
    fig2.clf()
    ymax = max([max(the_slice[ink_pt_x]), max(the_slice[no_ink_pt_x])]) + 1000

    #figure 1, subplot 0: slice image
    p1 = fig2.add_subplot(311)
    p1.imshow(the_slice,cmap="Greys_r")
    height = len(the_slice[0])
    p1.plot([0,height],[ink_pt_x,ink_pt_x],color='g',linewidth=2)
    p1.plot([0,height],[no_ink_pt_x,no_ink_pt_x],color='r',linewidth=2)



    #figure 2, subplot 1: ink plot
    g1 = fig2.add_subplot(312,sharex=p1)
    axes = fig2.gca()
    axes.set_ylim([0,ymax])
    g1.plot(the_slice[ink_pt_x],color='g',marker='o')
    if(len(ink_vect_peaks) > 1):
        ink_vect_start = ink_vect_peaks[0]
        ink_vect_end = ink_vect_peaks[-1]
        p1.plot((ink_vect_start),(ink_pt_x),color='g',marker='*')
        p1.plot((ink_vect_end),(ink_pt_x),color='g',marker='*')
        g1.scatter([ink_vect_start,ink_vect_end], \
                [ink_vect[ink_vect_start],ink_vect[ink_vect_end]], \
                color='g',marker='*')

    #figure 2, subplot 2: no-ink plot
    g2 = fig2.add_subplot(313,sharex=p1,sharey=g1)
    g2.plot(the_slice[no_ink_pt_x],color='r',marker='o')
    if(len(no_ink_vect_peaks) > 1):
        no_ink_vect_start = no_ink_vect_peaks[0]
        no_ink_vect_end = no_ink_vect_peaks[-1]
        p1.plot((no_ink_vect_start),(no_ink_pt_x),color='r',marker='*')
        p1.plot((no_ink_vect_end),(no_ink_pt_x),color='r',marker='*')
        g2.scatter([no_ink_vect_start,no_ink_vect_end], \
                [no_ink_vect[no_ink_vect_start],no_ink_vect[no_ink_vect_end]], \
                color='r',marker='*')

    #show dat plot
    plt.show()



if __name__ == "__main__":
    main()
    


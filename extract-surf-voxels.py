'''
extract-surf-voxels.py
Walk through the fragment, grabbing surface voxels 
Output:
	output.txt	all the 3d start points
	output.tiff	a 2D picture with the values at all the start points
'''

__author__ = "Jack Bandy"
__email__ = "jgba225@g.uky.edu"


import tifffile as tiff
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import pickle
from scipy.signal import argrelmax
from os import mkdir



def main():
	NUM_VOX = 3
	THRESH = 21000

	data_path = "/Users/jack/dev/ink-id/small_fragment_data"
	output_path = data_path+"/surf-output-"+str(THRESH)
	try:
		mkdir(output_path)
	except E:
		print("output directory already exists")
	
	ref_photo = tiff.imread(data_path+"/registered/aligned-photo-contrast.tif")
	the_slice = tiff.imread(data_path+"/vertical_rotated_slices/slice0000.tif")
	print("loaded data")
	
	img_frnt = np.zeros((ref_photo.shape[0],ref_photo.shape[1]),dtype=np.uint16)
	img_back = np.zeros((ref_photo.shape[0],ref_photo.shape[1]),dtype=np.uint16)
	end_pts = []
	strt_pts = []
	print("initialized output")
	
	
	for i in range(ref_photo.shape[0]):
		#for i in range(500,600):
		the_slice_name = (data_path+"/vertical_rotated_slices/slice" \
				+"0000"[:4-len(str(i))] + str(i)+".tif")
		skel_slice_name = (data_path+"/skeleton_slices/slice" \
				+"0000"[:4-len(str(i))] + str(i)+".tif")
				
		the_slice = tiff.imread(the_slice_name)
		skel_slice = np.zeros((the_slice.shape[0],the_slice.shape[1]),dtype=np.uint16)

		#for each vector in the slice
		#filter out everything beneath the threshold
		thresh_mat = np.where(the_slice > THRESH,the_slice,0)
		
		
		for v in range(the_slice.shape[0]):
			#vect = the_slice[v]
			#filter out everything beneath the threshold
			frag_width = np.where(thresh_mat[v] > 0)
			#find relative maximas
			#vect_peaks = argrelmax(vect)[0]
			if (len(frag_width[0]) > 1):
				start = frag_width[0][0]
				end = frag_width[0][-1]
				skel_slice[v][start] = the_slice[v][start]
				skel_slice[v][end] = the_slice[v][end]
				strt_pts.append((i,v,start))
				end_pts.append((i,v,end))
				img_frnt[i][v] = np.average(the_slice[v][start:start+NUM_VOX])
				img_back[i][v] = np.average(the_slice[v][end-NUM_VOX:end])
		

		if(i % (ref_photo.shape[0] / 20) == 0):
			print("{0} slices complete out of {1}".format(i,ref_photo.shape[0])) 
		tiff.imsave(skel_slice_name,skel_slice)
		#print("finished slice " + str(i))
	
	
	print("outputting data")
    
	tiff.imsave(output_path+"/front.tif",img_frnt)
	tiff.imsave(output_path+"/back.tif",img_back)
	np.savetxt(output_path+"/end_pts.txt",np.array(end_pts),fmt="%u",delimiter=",")
	np.savetxt(output_path+"/start_pts.txt",np.array(strt_pts),fmt="%u",delimiter=",")
	np.savetxt(output_path+"/img_back.txt",img_back,fmt="%u",delimiter=",")
	np.savetxt(output_path+"/img_front.txt",img_frnt,fmt="%u",delimiter=",")
	
	print("data outputted")








if __name__ == "__main__":
    main()
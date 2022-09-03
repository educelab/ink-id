#!/bin/bash

#### 
####
# This script needs to be run in the directory that contains the dataset directory
# (e.g. CarbonPhantom) and the directory must have the following descendant directory
# structure
#
# dataset/subvol_type/col/ground_truth/sample_num/saved_weights
# 
# Example:
# CarbonPhantom/NearestNeighbor/Col6/CarbonInk/0/m1252444
#
######
dataset=
subvol_type=
col=
ground_truth=
sample_num=
saved_weights=
prefix=$(pwd)

data_dir="${prefix}/${dataset}/${subvol_type}/${col}/${ground_truth}/${sample_num}/${saved_weights}"


#######
# Combine all the png files for each filter activations and save the pdf file
# inside the activations directory
#######

activations_dir="${data_dir}/IntermediateActivations"

function combine_images() {
# $1: filter name (also assumes it is the directory name)
# $2: number of images
# $3: output filename 

  montage -tile 4x -geometry +2+2 -title "${activations_dir}_${1}" \
  $(eval echo "${activations_dir}/${1}/${1}_{0..${2}}.png") ${activations_dir}/${3}; 
}


combine_images  conv1_activations 31 conv1_activations.pdf
combine_images  conv2_activations 15 conv2_activations.pdf
combine_images conv3_activations 7 conv3_activations.pdf
combine_images conv4_activations 3 conv4_activations.pdf

combine_images batch_norm1_activations 31 batch_norm1_activations.pdf
combine_images batch_norm2_activations 15 batch_norm2_activations.pdf
combine_images batch_norm3_activations 7 batch_norm3_activations.pdf
combine_images batch_norm4_activations 3 batch_norm4_activations.pdf

######
# Next combine those files
######

input_images_dir="${data_dir}/InputImages"

####
# Combine input-images into a input_images.pdf
####
convert  ${input_images_dir}/plotly.png \
	 ${input_images_dir}/tiff_slices.png \
	 ${input_images_dir}/slices.png \
	 ${input_images_dir}/input_images.pdf;

### 
# Combine filter images
#####
convert ${activations_dir}/conv4_activations/filter_0_slices.png \
        ${activations_dir}/conv4_activations/filter_1_slices.png \
        ${activations_dir}/conv4_activations/filter_2_slices.png \
        ${activations_dir}/conv4_activations/filter_3_slices.png \
	${activations_dir}/conv4_filter_slices.pdf;


convert ${activations_dir}/batch_norm4_activations/filter_0_slices.png \
        ${activations_dir}/batch_norm4_activations/filter_1_slices.png \
        ${activations_dir}/batch_norm4_activations/filter_2_slices.png \
        ${activations_dir}/batch_norm4_activations/filter_3_slices.png \
	${activations_dir}/batch_norm4_filter_slices.pdf;

####
# Convert json into a pdf
#####
#
enscript ${data_dir}/metadata.json --output=- | ps2pdf - > ${data_dir}/metadata.pdf;
#
#
#### 
# Finally, combine everything!
####
#

pdfunite \
  ${data_dir}/metadata.pdf \
  ${input_images_dir}/input_images.pdf \
  ${activations_dir}/conv1_activations.pdf  \
  ${activations_dir}/conv2_activations.pdf  \
  ${activations_dir}/conv3_activations.pdf  \
  ${activations_dir}/conv4_activations.pdf  \
  ${activations_dir}/conv4_filter_slices.pdf \
  ${activations_dir}/batch_norm1_activations.pdf  \
  ${activations_dir}/batch_norm2_activations.pdf  \
  ${activations_dir}/batch_norm3_activations.pdf  \
  ${activations_dir}/batch_norm4_activations.pdf  \
  ${activations_dir}/batch_norm4_filter_slices.pdf \
  ${data_dir}/final_result.pdf;




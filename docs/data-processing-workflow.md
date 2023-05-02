# Before starting

Command-line applications from Volume Cartographer and registration-toolkit are shown being run by name in this document. If these applications have not been installed to your system path, you must modify the provided examples to point to the location of your compiled applications, often the `build/bin` folder for each project. For example, you might do the following to run `vc_render` from the `volume-cartographer` folder in your home directory:

```shell
~/volume-cartographer/build/bin/vc_render ...
```

You can temporarily add the `bin/` directory from `volume-cartographer` to your path using the following:

```shell
export PATH=$PATH:~/volume-cartographer/build/bin/
```

# Inventory and get working copy on LCC

Files may be distributed to begin with. Check the following documents and locations to make an inventory of what work has been done on this object and where it is currently stored. If necessary, merge the work done into one central location. The preferred central location is `gemini1-2:/mnt/gemini1-4/seales_uksr/nxstorage/data/` in a directory structure such as (example) `Herculaneum_Scrolls/PHercParis2/Frag47/PHercParis2Fr47.volpkg`. However, `gemini` is only usable for storage and not active work. So start by consolidating things on `lcc:/pscratch/seales_uksr/` or your local machine. The former may be necessary due to disk space.

* [Moonshot Data Progress Tracking Sheet](https://docs.google.com/spreadsheets/d/16s8GkQ74w5fmp6d1MwYGtmcf26gk9PjrD_ldManLhKw/edit?usp=sharing) (old)
* [CT Data Manifest](https://luky-my.sharepoint.com/:x:/r/personal/cpa232_uky_edu/Documents/Projects/2022/20221223%20-%20CT%20Manifest/CT%20Data%20Manifest.xlsx?d=w63135ef37bd7489d8314b1cad1bc218c&csf=1&web=1&e=VSUpkm) (new)
* `gemini1-1:/mnt/gemini1-3/seales_uksr/`
* `gemini1-2:/mnt/gemini1-4/seales_uksr/`
* `lcc:/pscratch/seales_uksr/`

# Determine crop

It is necessary to load the volume into a visualizer such as ImageJ to determine the crop bounding box. If the slices are already generated and on your local disk in one volume, which is likely if they came from a benchtop machine, those can be viewed directly. With the large .hdf files from Diamond, it is easiest to first extract some sample slices on LCC, transfer those to your machine and then view them there. Those can be extracted all at once from one or more .hdfs (multiple desirable if object scanned in slabs) like this:

```shell
sbatch -p CAL48M192_L --time=14-00:00:00 inkid_general_cpu.sh \
python /usr/local/dri/ink-id/inkid/scripts/hdf_extract_slices.py \
--input-files /pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis2/Frag47/CT/54keV/row_000_scan_91865_91866/full_recon_row_000_scan_91865_91866.hdf \
/pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis2/Frag47/CT/54keV/row_001_scan_91867_91868/full_recon_row_001_scan_91867_91868.hdf \
/pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis2/Frag47/CT/54keV/row_002_scan_91869_91870/full_recon_row_002_scan_91869_91870.hdf \
/pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis2/Frag47/CT/54keV/row_003_scan_91871_91872/full_recon_row_003_scan_91871_91872.hdf \
/pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis2/Frag47/CT/54keV/row_004_scan_91873_91874/full_recon_row_004_scan_91873_91874.hdf \
/pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis2/Frag47/CT/54keV/row_005_scan_91875_91876/full_recon_row_005_scan_91875_91876.hdf \
--dataset-name entry/data/data \
--auto-percentile-windowing \
--output-dir /pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis2/Frag47/CT/54keV/2023-02-04_sample_every_100_slices/ \
--combine-output-in-single-dir \
--slice-skip 100
```

The next step is the same for either the original full set of slices (if that is feasible and on your machine) or the subset sampled using something like the above.

Load the volume in Fiji/ImageJ, and select a rectangular bounding box. Scrub through the slices, adjusting the bounding box to make sure it is always outside the bounds of the object of interest. It is good to crop tightly to create smaller datasets, however it is important to be sure to contain the entire object. When in doubt, it's OK to make the bounding box a little bigger to be sure it includes the object.

Record the x and y offsets as well as the width and height of the bounding box you have selected. You can either mouse over the edges and get close estimates from viewing the mouse coordinates in the ImageJ main window, or you can use `Analyze -> Tools -> ROI Manager`, add the current ROI, and use `More -> Specify...` withing the ROI manager to see the bounding box specs.

![Screenshot_2023-02-04_at_3.29.53_PM](uploads/1611b662833a0ae3f76a537258c666d9/Screenshot_2023-02-04_at_3.29.53_PM.png)

# Generate/extract slices, window, and crop x-y

The goal of this step is to get a 16-bit .tif image stack. For benchtop sources, this is probably already done as part of the reconstruction process. For synchrotron scans, this step may be necessary. For example in the 2019 Diamond Light Source scans, the reconstruction output 32-bit float .hdf files from which .tif slices need to be extracted. Often, such as with the fragments scanned in that session, there is a separate .hdf for each "slab". Slices should be extracted from each, and then merged later. The extraction for multiple .hdf files can be done in one command.

The range of values in the float .hdf is not the same as the 16-bit integer representation, so the values need to be stretched to \[0-65535\] during this process. Use `--auto-percentile-windowing` to do this automatically. For intact scrolls, where much more of the scan is papyrus, use `--percentile-min 0.1 --percentile-max 99.9`. For fragments, where less of the scan is papyrus and so less clipping should happen, use the defaults (`--percentile-min 0.01 --percentile-max 99.99`, but you don't need to specify this).

For particularly large datasets (such as those split into slabs) the entire dataset may not fit on your desktop machine. In these cases it may then be more efficient to crop the source files on the source server before transferring them to your desktop for tasks requiring a graphical interface/user intervention. This allows for slabs to be processed in parallel up until volume packaging and greatly reduces the size of the initial data transfer.

Using the crop boundaries determined in the previous step, here is an example command continuing the process for the same data:

```shell
sbatch -p CAL48M192_L --time=14-00:00:00 inkid_general_cpu.sh python /usr/local/dri/ink-id/inkid/scripts/hdf_extract_slices.py \
--input-files /pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis2/Frag47/CT/54keV/row_000_scan_91865_91866/full_recon_row_000_scan_91865_91866.hdf \
/pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis2/Frag47/CT/54keV/row_001_scan_91867_91868/full_recon_row_001_scan_91867_91868.hdf \
/pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis2/Frag47/CT/54keV/row_002_scan_91869_91870/full_recon_row_002_scan_91869_91870.hdf \
/pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis2/Frag47/CT/54keV/row_003_scan_91871_91872/full_recon_row_003_scan_91871_91872.hdf \
/pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis2/Frag47/CT/54keV/row_004_scan_91873_91874/full_recon_row_004_scan_91873_91874.hdf \
/pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis2/Frag47/CT/54keV/row_005_scan_91875_91876/full_recon_row_005_scan_91875_91876.hdf \
--dataset-name entry/data/data \
--auto-percentile-windowing \
--output-dir /pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis2/Frag47/CT/54keV/2023-02-04_slices/ \
--crop-min-x 534 \
--crop-width 7198 \
--crop-min-y 4555 \
--crop-height 1399
```

# Merge slices, crop z

If the slices come from individual slabs, they must be merged together into a single volume.

First, the offsets must be determined.
Starting with the overlap between slab 000 and slab 001, load both as virtual image stacks in ImageJ.
Set slab 000 to slice 0, and then scrub through the slices of slab 001 until finding the one that most resembles slab 000, slice 0.
The offsets are typically quite uniform, so you can likely expect it to be around slice 1548.
Record the individual slice number that best appears to be the overlap slice based on visual judgement.
This is primarily a visual sanity check; the actual best slice will be computed.
Take a note like this:

```
PHercParis2Fr47_54keV_slab_000 slice 0000 matches PHercParis2Fr47_54keV_slab_001 slice 1547
```

Then, confirm those results using `inkid/scripts/find_vertical_overlap.py`. For example:

```
python inkid/scripts/find_vertical_overlap.py --fixed-slab ~/data/dri-datasets-drive/PHerc2Fr47/slab_000 --moving-slab ~/data/dri-datasets-drive/PHerc2Fr47/slab_001 --min-index 1540 --max-index 1560 --num-candidates 50 --comparison-function pearson
```

This will compare at most 50 slices (which is more than the suggested range, so this argument is effectively ignored) between slices 1540 and 1560 of slab 001, using the Pearson correlation coefficient to compare each against slice 0 of slab 000.
A sorted list of slice numbers is printed with their associated correlation.
The most highly correlated slice should be at the bottom.
Verify it is at least close (typically within 2-3, ideally within 0-1 slices) to the manual guess.

Repeat this for each pair of adjacent slabs, resulting in a set of notes like this:

```
PHercParis2Fr47_54keV_slab_000 slice 0000 matches PHercParis2Fr47_54keV_slab_001 slice 1547 | script agrees
PHercParis2Fr47_54keV_slab_001 slice 0000 matches PHercParis2Fr47_54keV_slab_002 slice 1550 | script says 1549
PHercParis2Fr47_54keV_slab_002 slice 0000 matches PHercParis2Fr47_54keV_slab_003 slice 1547 | script says 1548
PHercParis2Fr47_54keV_slab_003 slice 0000 matches PHercParis2Fr47_54keV_slab_004 slice 1549 | script agrees
PHercParis2Fr47_54keV_slab_004 slice 0000 matches PHercParis2Fr47_54keV_slab_005 slice 1548 | script says 1549
```

Then, the merge itself can be performed using `merge_slabs.py`.
For arbitrary sets being merged, care should be taken regarding the ordering of the slabs and the ordering of the slices within the slabs.
Check the documentation inside `merge_slabs.py` for more information if this is necessary.
Specifically for 2019 fragment scans from Diamond Light Source, the slice indices can be passed in the order they are listed above, and `--reverse-slab-order` should be supplied.

```
sbatch -p CAL48M192_L --time=14-00:00:00 inkid_general_cpu.sh python /usr/local/dri/ink-id/inkid/scripts/merge_slabs.py --in-dir /pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis2/Frag47/CT/54keV/2023-02-04_slices/ --out-dir /pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis2/Frag47/CT/54keV/2023-02-05_merged_slices --indices 1547 1549 1548 1549 1549 --reverse-slab-order
Submitted batch job 1502493
Taking slices 0 to 1548 from slab full_recon_row_005_scan_91875_91876
Taking slices 0 to 1548 from slab full_recon_row_004_scan_91873_91874
Taking slices 0 to 1547 from slab full_recon_row_003_scan_91871_91872
Taking slices 0 to 1548 from slab full_recon_row_002_scan_91869_91870
Taking slices 0 to 1546 from slab full_recon_row_001_scan_91867_91868
Taking slices 0 to 2149 from slab full_recon_row_000_scan_91865_91866
```

Following this process, there will be a merged volume in the output directory of the above script.
Load it as a virtual image stack in ImageJ and confirm the merged volume appears coherent and does not have jumps between slabs.
Also note the approximate beginning slice and ending slice of the fragment itself, if it is smaller than the entire slice range.
The slices before and after the fragment can then be "cropped" (deleted) like this, further reducing the data size:

```
rm {0000..1330}.tif  # Remove slices before fragment
rm {8550..9891}.tif  # Remove slices after fragment
```

# Import slices into .volpkg

Use `vc_packager` to import the slices into a volume package.
The following flags will be required:

* `--volpkg (-v)`: The path and name of the volume package
* `--slices (-s)`: Path to an importable volume.

If the object does not already have a volume package, the following will also be required:

* `--material-thickness (-m)`: The estimated thickness of a layer (i.e. page) in microns

For Herculaneum scans, use 150 as the estimated material thickness.

You will be prompted for a descriptive name for the volume.
Base the name on this example: `PHercParis2Fr47 88keV merged, cropped 7332x1608x7229+552+4596+1331`, where the numbers following `cropped` are `crop-width`, `crop-height`, `crop-slices`, `x-min`, `y-min`, `z-min`.
You will also input the voxel size of the volume in microns, which for 2019 Diamond Light Source fragment scans is 3.24.
Finally you will choose whether to flip the images (for 2019 Diamond fragment scans, type `all`), and whether to compress the resulting slice images (keep the default, which is not to compress).
This entire process should look something like this:

```
~/src/volume-cartographer/build/bin/vc_packager -v . -s ~/temp/PHercParis2Fr47_88keV_merged_slices/
Getting info for Volume: "~/temp/PHercParis2Fr47_88keV_merged_slices/"
Enter a descriptive name for the volume: PHercParis2Fr47 88keV merged, cropped 7332x1608x7229+552+4596+1331  
Enter the voxel size of the volume in microns (e.g. 13.546): 3.24
Flip options: Vertical flip (vf), horizontal flip (hf), both, z-flip (zf), all, [none] : all
Compress slice images? [yN]:
Adding Volume: "~/temp/PHercParis2Fr47_88keV_merged_slices/"
```

Other `vc_packager` examples:

```shell
# Create a new Volume Package
vc_packager -v PHerc2Fr47.volpkg -m 150 -s PHerc2Fr47/volumes/54kV_cropped/

# Add to an existing Volume Package
vc_packager -v PHerc2Fr47.volpkg -s PHerc2Fr47/volumes/88kV_cropped/
```

`vc_packager` supports importing Grayscale or RGB images in the TIFF, JPG, PNG, or BMP formats with 8 or 16 bits-per-channel. The `--slices` option accepts a number of different importable volume types:

* **Path to a Skyscan reconstruction log file**: If the dataset is a reconstruction from a Skyscan scanner, provide the path to the Skyscan reconstruction log file (`.log`). In addition to adding the slices to the Volume Package, `vc_render` will also automatically import associated scan metadata, such as the voxel size.
* **printf-style file pattern for slice images**: If a directory of slice images also contains images that are not slices, specify the file pattern for the slice images using a printf-style pattern. Currently, only `%d%` and `%00d` replacements are supported.
* **Path to a directory of slices**: If a directory contains only slice images, simply specify the path to the directory. Image detection is greedy, and any images in the directory that are not slice images will cause errors. Note that lexicographical sort is used to arrange the images, therefore it is a good idea to zero-pad any numerical orderings in the image filenames.

```shell
# Skyscan reconstruction directory
vc_packager -v ObjectName.volpkg -s Skyscan_rec/Skyscan_rec.log

# File name pattern
# Matches slices_0000.tif, slices_0001.tif, slices_0002.tif, ...
vc_packager -v ObjectName.volpkg -s MixedData/slices_%04d.tif

# Directory of slice images
vc_packager -v ObjectName.volpkg -s OnlySlices/
```

# Make slice video

Use `ffmpeg` to make a video of the volume slices.
Note that the input tif files specified using `-i` must match the number of digits in the slice filenames (usually `%04d.tif` or `%05d.tif`).

Example command:

```shell
ffmpeg -r 30 -f image2 -i /pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis4/PHercParis4.volpkg/volumes/20230210143520/%05d.tif -vf "scale=3840:-1,pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx264 -crf 32 -pix_fmt yuv420p /pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis4/PHercParis4.volpkg/working/PHercParis4_54keV_stitched_part_1.mp4
```

The same command can be run on LCC using the VC Singularity container defined by `ink-id/inkid/scripts/singularity/vc.def`.
As with many of the LCC commands in this process, it is helpful to request all memory on the node using `--mem=0`.

```shell
sbatch -p SKY32M192_L --mem=0 --time=5-00:00:00 vc_general_cpu.sh ffmpeg -r 30 -f image2 -i /pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis4/PHercParis4.volpkg/volumes/20230210143520/%05d.tif -vf "scale=3840:-1,pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx264 -crf 32 -pix_fmt yuv420p /pscratch/seales_uksr/nxstorage_partial_copy_in_case_gemini_down/data/Herculaneum_Scrolls/PHercParis4/PHercParis4.volpkg/working/PHercParis4_54keV_stitched_part_1.mp4
```

The slice video should be placed in the `working/` directory of the `.volpkg`.
Watch the slice video through, checking that the object of interest stays within the crop for all slices, and that there is not any visible data corruption.

# Segmentation

## Hidden layers

Now two options are available for segmenting hidden layers:

1: For hidden layers or wrapped objects, you can use the VC GUI app to perform segmentation. This will create a new Segmentation directory in the `paths` subdirectory of the Volume Package. Make sure to add the Segmentation ID to the progress tracker spreadsheet.

2: A new segmentation approach exists, currently referred to as "Thinned Flood Fill Segmentation". This algorithm is not incorporated into the VC GUI app at this time. However, this algorithm is very useful for quickly segmenting layers of manuscripts where separation between layers is obvious in the tomographic data. The instructions for using Semi-Automated Segmentation in its current form are outlined below:

* First, create a directory to hold your segmentations. All volume packages should have a working directory that would be a good place to create a subdirectory to hold your segmentations.
* Open the VC GUI app and load the volume package you want to segment from. Start a new segmentation and place the points along the layer you want to segment. Make sure to disable the Pen Tool when you are finished so the points are saved.
* Next, open a terminal window, navigate to the directory you created, and execute vc_segment. A template for this is provided below. The segmentation files will be saved in the current working directory. Be careful not to overwrite the data you created in previous executions.
  * dump-vis, save-mask, save-interval (set at 1) are optional, but strongly recommended parameters.
  * -l is the low threshold parameter. It takes a 16-bit grayscale value (0-65535). For M.910, a value near 13000-14000 is typically a good choice.
  * tff-dt-thresh (a float value, ranging from 0.-1.) determines how much of the mask will be pruned away before the thinning algorithm begins. Setting this too high will disrupt the continuity of the skeleton. Setting it such that the continuity is not disrupted is important. Leaving this parameter out will result in none of the mask being pruned away, so this is a good option if the layer is extremely thin. This parameter can be useful if set correctly, as some segmentation errors will be eroded away.

```
path/to/volume-cartographer/build/bin/vc_segment -m TFF -s VC_SEGMENTATION_ID -v /path/to/MS910.volpkg/ --start-index START_SLICE --end-index END_SLICE -l LOW_GRAY_THRESHOLD_16_BIT_GREY --tff-dt-thresh THRESHOLD --dump-vis --save-mask --save-interval 1
```

(Fall 2020) M.910 Common Settings:

```
path/to/volume-cartographer/build/bin/vc_segment -m TFF -s VC_SEGMENTATION_ID -v /path/to/MS910.volpkg/ --start-index START_SLICE --end-index 5480 -l 13500 --tff-dt-thresh .2 --dump-vis --save-mask --save-interval 1
```

* If you enabled dump-vis, a new directory called 'debugvis' will appear inside the working directory. Two directories inside, called 'mask' and 'skeleton', contain images that show what is being segmented. You can use these images as a reference to help you determine when to stop the segmentation.

To obtain a good-quality segmentation, the mask must cover the majority of the layer of interest, but it is fine if some small parts aren't covered or parts of neighboring pages get segmented too. This is an example of a good-quality segmentation: https://drive.google.com/file/d/1_qzL2L2gZpYHYUJznCZENbsW2ueUj8\\\\\\\\\\\\\\\_\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_/view?usp=sharing

Check the segmentation occasionally. If a lot of neighboring pages are getting segmented or if the segmentation loses the layer you are segmenting, use Ctrl-C to kill vc_segment **provided you ran vc_segment with `--save interval 1`**.

* Save all of the files outputted by vc_segment into a subdirectory. (For example, if you are segmenting page 20, you could name the directory 20_1 if it was your first segmentation of that page. Just be consistent.)
* Go back to the VC GUI app and create another _new_ segmentation (do not try to re-use the old one; VC will crash) at the first slice that the previous segmentation did badly on. Run vc_segment with the new segmentation, starting at the new segmentation's slice number. Repeat this process of running, stopping, and resetting until the page is segmented as much as possible (near the end of the volume it may become very difficult to keep going (for M.910, this can happen when you are in/near the 4000s) so just stop if you are near the end of the volume and it is too difficult.)
* Merge the vcps files to make a single large file. Do this by using vc_merge_pointsets. Usage:

```
path/to/volume-cartographer/build/bin/vc_merge_pointsets -i path/to/dir_containing_all_pointsets -o path/to/output_dir
```

Note: vc_merge_pointsets takes a directory that contains all vcps files you wish to merge as input. Put all vcps files in there. You will need to rename them, but make sure to keep the .vcps extension. Note: Use the --prune flag to prune the pointsets during the merge. **Important: every vcps file must be named the last slice number of that segmentation that you wish to keep (example: a segmentation that goes to 1200 slices but is only valid until 1150 should be named 1150.vcps)** (Coming soon)

* Convert the vcps file into a point cloud. Do this by using vc_convert_pointset. Usage:

```
path/to/volume-cartographer/build/bin/vc_convert_pointset -i pointset_name.vcps -o output_mesh_name.obj
```

- (Fall 2020) Upload the merged point cloud and all pointsets (mask_pointset.vcps and pointset.vcps for each segmentation) to the DRI Experiments Shared Drive inside the folder linked here: https://drive.google.com/drive/folders/1U7wg1mGDlg6wLsRx_EtCIEMsNmh8yxJj?usp=sharing Create a new folder named after the page number, and put the .vcps files, point clouds, and meshes inside that folder.
- **(Fall 2020: ignore this step)** These point clouds need further cleaning before they can be turned into a mesh. This process is done in Meshlab. Refer to the processing instructions for canny segmentation below (see the "Exposed layers" section.) The same steps will be necessary for these point clouds.

## Exposed layers

For flat, exposed layers, manually segment the layer using the canny edge detection segmentation utility. To better keep track of these manually define segmentations, first make a working directory for your segmentation inside the Volume Package and then run `vc_canny_segment`:

```shell
# Make a working directory inside the volpkg
cd PHerc2Fr143.volpkg/
mkdir -p working/54kv_surface_layer/
cd working/54kv_surface_layer/

# Run canny segment
# Note: ../../ is the path to PHerc2Fr143.volpkg/
vc_canny_segment -v ../../ --volume 20200125113143 --visualize -o canny_raw.ply
```

For each slice in the specified volume, `vc_canny_segment` runs a canny edge detector to isolate surfaces. It then marches along each row or column of the image and projects a line from an edge of the image onto the canny-detected edges. By default, the first edge detected becomes an 3D point in the output point set. This utility has a number of useful options for controlling the segmentation:

* `--projection-edge (-e)`: The edge to project from when detecting surface points. Accepts: `L, R, T, B` for Left, Right, Top, and Bottom respectively.
* `--visualize`: Opens a window that lets you preview the canny edge detection options. Adjust the parameters until the edge of the fragment you are trying to segment is highlighted in white, but any background noise is not highlighted. Make sure the min threshold is less than the max threshold. If they are flipped, the finished segmentation may not look as expected. Once done, hit any key to run the segmentation.
* `--mask`: Provide a B&W image where white indicates the region of the volume to consider for edge detection. Any canny edges in the slices that overlap with the black portion of the mask will be ignored.

The output of this process, `canny_raw.ply`, is a dense point set and requires further processing in Meshlab: 0. Select `View -> Toggle Orthographic Camera` in Meshlab to use the orthographic camera, which removes perspective distortion when viewing the point cloud and makes it easier to select regions manually.

 1. Run `Filters/Point Set/Point Cloud Simplification` to reduce the point set to a reasonable size. If the surface is very smooth, use fewer points. Usually, within the order of 10k to 100k points typically retains enough detail while significantly speeding up later steps. Save this point set with the name: `01_simplified.ply`.
 2. Manually select and delete points that are not on the desired surface.
 3. Run `Filters/Selection/Select Outliers` and then delete the selected vertices. This cleans up groups of points that are not on the surface. It is recommended to enable the Preview option while tuning the selection options.
 4. Run `Filters/Point Set/Compute normals for point sets` to estimate surface normals for the point set. For dense, noisy point sets, adjust the `Neighbor num` value to a larger value, typically no more than 100. Save this point set to your working directory with the name: `canny_cleaned.ply`
 5. Run `Filters/Remeshing, Simplification and Reconstruction/Surface Reconstruction: Screened Poisson` to triangulate the surface. This filter uses the surface normals generated in the previous step to fit a continuous surface to the point set. Increase the Reconstruction depth to make the surface fit more closely to the original point set at the expense of more faces and a rougher surface. Typically, use a reconstruction depth in the range of 8-10. Save this mesh to your working directory with the name: `canny_poisson.ply`
 6. Poisson will create faces which extend beyond the original point set. Run `Filters/Sampling/Hausdorff Distance` to add an attribute to each vertex of the new surface that is that vertex's distance to the nearest point in the original point set.
 7. Run `Filters/Selection/Select by Vertex Quality` to select those vertices in the Poisson surface which have large distances from the original point set. Use the Preview option to tune the selection. Delete the selected vertices and faces.
 8. Run `Filters/Cleaning and Repairing/Remove T-Vertices by Edge Flip`. The number of removed t-vertices will be reported in the log panel.
 9. Run `Filters/Quality Measure and Computations/Compute Topological Measure`. A topology report will be printed in the log panel in the bottom-right of Meshlab. Based on what is reported, perform the following steps. Repeat this step as needed.
    * `Unreferenced Vertices N > 0`: Run `Filters/Cleaning and Repairing/Remove Unreferenced Vertices`. The number of removed vertices will be reported in the log panel.
    * `Mesh is composed by N > 1 connected component(s)`: Run `Filters/Cleaning and Repairing/Remove Isolated pieces (wrt Face Num.)` to remove small, disconnected connected components. Use a component size of 1000 or larger to ensure that all surfaces that you will remove are not connected to your segmented surface. The number of removed connected components will be reported in the log panel.
    * `Mesh has N > 0 non two manifold edges...`: Run `Filters/Cleaning and Repairing/Repair non Manifold Edges by removing faces`.
    * `Mesh has N > 0 holes`: Run `Filters/Remeshing, Simplification and Reconstruction/Close Holes`. Adjust `Max size to be closed` to large values until all holes are closed.
10. Save your final mesh as a new file with a name which matches your working directory (e.g. `54kv_surface_layer.ply`). After selecting the output file location, a window with saving options will open. Click the box to uncheck `Binary encoding` to save the file in an ASCII format. **This is required for using this mesh with vc_render.**

# Texturing

All texturing should be performed with the `vc_render` command-line application. Do not use VC Texture.app.

## Segmentations from VC.app

Make a new working directory for your segmentation inside the Volume Package and provide `vc_render` with the volume package and segmentation ID of your segmentation:

```shell
# Make a working directory inside the volpkg
cd PHerc2Fr143.volpkg/
mkdir -p working/54kv_internal_layer/
cd working/54kv_internal_layer/

# Run vc_render
# Note: ../../ is the path to PHerc2Fr143.volpkg/
vc_render -v ../../ -s 20200125113143 --output-ppm 54kv_internal_layer.ppm --uv-plot 54kv_internal_layer_uvs.png --method 1 -o 54kv_internal_layer.obj
```

## Segmentations from canny segmentation

Provide `vc_render` with the volume package, the final mesh produced by Meshlab, and the ID of the segmented volume:

```shell
# Run vc_render
# Note: We are in working/54kv_surface_layer/
# ../../ is the path to PHerc2Fr143.volpkg/
vc_render -v ../../ --input-mesh 54kv_surface_layer.ply --volume 20200125113143 --output-ppm 54kv_surface_layer.ppm --uv-algorithm 2 --uv-plot 54kv_surface_layer_uvs.png --method 1 -o 54kv_surface_layer.obj
```

## Retexturing the segmentation

The above commands generate a texture image using the Intersection texture method (`--method 1`). This is the fastest texturing method and will help you more quickly verify that your flattened surface is correctly oriented and contains no significant flattening errors. However, this image is not always useful for aligning the reference image. If you have difficulty finding point correspondences in the [registration step](#align-the-reference-image), use the `vc_render_from_ppm` utility to generate new texture images using alternative parameters:

```shell
vc_render_from_ppm -v ../../ -p 54kv_surface_layer.ppm --volume 20200125113143 -o 54kv_surface_layer_max.png
```

There are many texturing parameters available in both `vc_render` and `vc_render_from_ppm`. We have found that the following alternatives are consistently useful for generating new textures:

* Composite method, Max filter: The default texturing method if no options are passed. Returns the brightest intensity value in the neighborhood. Enable with these options: `--method 0 --filter 1`
* Composite method, Mean filter: Return the average of the neighborhood's intensity values. Useful if the dataset is noisy. Enable with these options: `--method 0 --filter 3`
* Integral method: Return the sum of the neighborhood's intensity values. Sometimes shows subtle details that are missed by the Composite method. Enable with these options: `--method 2`.
* Adjust the texturing radius: The size of the texturing neighborhood is automatically determined by the Volume Package's material thickness metadata field. Because this value is an estimate of a layer's thickness, it is sometimes too small/too large. To manually set the search radius, provide the `--radius` option a real value in voxel units.

## Speeding up flattening and PPM generation

The processing times for flattening and PPM generation are sensitive to the number of faces in the segmentation mesh. In particular, meshes generated from the `vc_canny_segment` process are often densely sampled, thus leading to long processing times. For these meshes, use the mesh resampling options in `vc_render`:

```shell
vc_render -v ../../ --input-mesh 54kv_surface_layer.ply --volume 20200125113143 --enable-mesh-resampling
```

See `vc_render --help` for more options related to resampling. This flag is enabled by default for segmentation inputs passed with the `-s` option, but disabled for all inputs pass with `--input-mesh`. The number of vertices in the output mesh can be controlled with the `--mesh-resample-factor` option, which sets the approximate number of vertices per square millimeter in the resampled mesh. Newer versions of volume-cartographer (5abb42db and up) additionally have the `--mesh-resample-vcount` option which exactly controls the number of vertices in the output mesh. Be careful to not set the vertex count value too low, as this can modify your mesh such that it no longer intersects the object's surface.

## Fixing orientation errors

The various flattening (aka UV) algorithms available in volume-cartographer will produce flattened surfaces which are often flipped or rotated relative to what the observer would expect if they were to look at the surface in real life. The presence of these transformations may not become known until attempting to align the reference photograph to the generated texture image. **Textures, PPMs, and all subsequent steps should be updated to match the expected orientation when these problems are detected.**

The `vc_render` application provides the `--uv-rotate` and `--uv-flip` options to adjust for these transformations. The effect of these flags can be previewed without waiting to generate a full texture image by looking at the file specified by the `--uv-plot` flag:

```shell
# Rotate the UV map 45 degrees
# 54kv_internal_layer_uvs.png will be updated immediately after flattening and before PPM generation
vc_render -v ../../ -s 20200125113143 --output-ppm 54kv_internal_layer.ppm --uv-plot 54kv_internal_layer_uvs.png -o 54kv_internal_layer.obj --uv-rotate 45

# Flip the UV map horizontally
vc_render -v ../../ -s 20200125113143 --output-ppm 54kv_internal_layer.ppm --uv-plot 54kv_internal_layer_uvs.png -o 54kv_internal_layer.obj --uv-flip 1
```

Consult an expert or scholar to ensure the orientation at this stage is correct. To us CS folk, it can be easy to have text that looks correct but is actually mirrored, for example. This is the time to make sure it is oriented correctly!

# Align the reference image

## Using algorithmic registration

Using the [Landmark Picker GUI app](https://gitlab.com/educelab/landmark-picker), generate a landmarks file which maps points in the reference photograph onto the same points in the texture image generated by the previous step.

- Load the texture image as the Fixed image and the reference photograph as the Moving image.
- Select 6+ point correspondences between these two images.
- Export the landmarks file to your working directory: `54kv_surface_layer_landmarks_ref2ppm.ldm`

Run registration using the `rt_register2d` application, passing the texture image as the fixed input and the reference photograph as the moving input:

```shell
rt_register2d -f 54kv_surface_layer.png -m PHercParis2_Fr143r_IR940.jpg -o 54kv_surface_layer_photo.png -t 54kv_surface_layer_ref2ppm.tfm -l 54kv_surface_layer_landmarks_ref2ppm.ldm --disable-deformable
```

Any images which were previously aligned to the original moving image can additionally be aligned to the texture image using the generated transform file (`.tfm`) and `rt_apply_transform`:

```shell
rt_apply_transform -f 54kv_surface_layer.png -m PHercParis2_Fr143r_RGB.jpg -t 54kv_surface_layer_ref2ppm.tfm -o 54kv_surface_layer_photo_rgb.png 
```

This can be useful if you wish to align an RGB photograph to the texture image, but surface details can only be seen in an alternative channel (i.e. infrared).

## Manual registration using Photoshop's Puppet Warp

- drag files into PS
- will consolidate into render layer (need to take other layers back to 8 bit later)
- rename render layer "texture"
- go to photo, duplicate into texture, rename it "photo"
- right click on that, convert to smart object (.psb)
- double click on it to open it in a new tab
- go to ink labels, duplicate into that new psb
- name that layer ink-labels or something useful
- toggle that layer's visibility by clicking eyeball
- save that psb with command+S
- close ink label and photo image so they are out of the way
- save the original tab as a ps file (.psd)
- select the psb layer and go to edit->puppet warp
- in puppet warp you set pins, you can move the pins and it moves the pixels accordingly
- can play with opacity in layers thing
- can play with transfer function (dropdown that says "normal") in layers thing
- can turn "show mesh" off at the top
- when done open up smart layer (.psb), turn on ink-label visibility, and save. (this controls the way in which the smart layer appears in the main .psd file)
- go back to the main (.psd) file and turn the visibility on for both "photo" and "texture" layers. (otherwise, as a result of puppet-warping, the "photo" layer may no longer be a perfect rectangle any more)
- still on the main (.psd) file, choose File->Save a Copy, choose PNG as the format and save.
- open the saved .png file in Photoshop, go to image->mode and change it to "grayscale" "8-bit".
- repeat the previous 4 steps with photo visibility turned on (and saved) in the smart layer (.psb)
- can re-enter puppet warp and make more changes if desired by double clicking "puppet warp" under smart filters in layers dialog
- might want to read up on puppet warp documentation

# Generate ink labels

Ink labels are black-and-white images which indicate those areas of the PPM which contain ink and those which do not. They are manually created in Photoshop using the following steps:

* Open the aligned texture image in Photoshop.
* Use the [Quick Selection Tool](https://helpx.adobe.com/photoshop/using/making-quick-selections.html#select_with_the_quick_selection_tool) to select all regions of visible ink in the reference photograph (**Note:** By holding the Alt/Option key, you can easily switch between adding to/removing from the current selection).
* Once you are satisfied with your selection, click the `Create a new layer` button at the bottom of the Layers panel.
* With the new layer selected in the Layers panel, hit `Shift + Backspace/Delete`. In the dialog box which opens, set `Contents` to `White` and click OK.
* Hit `Ctrl/Cmd + D` to clear the pixel selection.
* Double-click the new layer name in the Layers panel and rename the new layer `Ink labels`.
* Click `Create a new layer` again and drag this new layer below the `Ink labels` one in the Layers panel.
* Rename this new layer to `Background`.
* With the `Background` layer selected, hit `Shift + Backspace/Delete`. In the dialog box which opens, set `Contents` to `Black` and click OK. The image displayed should now look like a binary pixel mask.
* Deselect all layers in the Layers panel. At the bottom of the panel, click `Create new fill or adjustment layer` and select the Threshold option. In the Threshold properties panel which opens, set the Threshold Level to `255`.
* Save the project as a Photoshop file (`.psd`) to your working directory (e.g. `54kv_surface_layer_inklabels.psd`)
* Select `Image/Mode/Grayscale`. When prompted, flatten the layers and discard the color information.
* Select `File/Save As...` and save this image as a PNG to your working directory (e.g. `54kv_surface_layer_inklabels.png`). **Be careful not to overwrite the Photoshop file saved previously.**
* Close Photoshop but **do not** save the Photoshop file.

# Region set

For now, manually create region set .json file defining training and prediction regions.

# Run ML

Based on inkid documentation and examples, found in the README here. The “SLURM Jobs” section points you to documentation for running jobs using SLURM and Singularity. A prebuilt container is available here so you shouldn’t have to go through the build process yourself.

import argparse
import json
import os

import numpy as np
from PIL import Image
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', metavar='DIR')
    args = parser.parse_args()

    # See if we can find a root of the DRI Datasets Drive folder locally
    dri_datasets_drive_path = None
    for candidate_path in [os.path.expanduser('~/data/dri-datasets-drive'),
                           '/pscratch/seales_uksr/dri-datasets-drive']:
        if os.path.exists(candidate_path):
            dri_datasets_drive_path = candidate_path
            break
    if dri_datasets_drive_path is None:
        print('Could not find a root directory for dri-datasets-drive')
        return

    # See if we can find a root of the invisible-library folder locally
    invisible_library_path = None
    for candidate_path in [os.path.expanduser('~/repos/invisible-library'),
                           '/pscratch/seales_uksr/invisible-library']:
        if os.path.exists(candidate_path):
            invisible_library_path = candidate_path
            break
    if invisible_library_path is None:
        print('Could not find a root directory for invisible-library')
        return

    # Look inside invisible-library for the Datasets folder
    datasets_dir = os.path.join(invisible_library_path, 'Datasets')
    if not os.path.exists(datasets_dir):
        print(f'Could not find Datasets directory inside invisible-library '
              f'directory: {invisible_library_path}')
        return

    # Find which tasks we have results for
    task_dirs = [os.path.join(args.results, f) for f in os.listdir(args.results)]
    task_dirs = list(filter(os.path.isdir, task_dirs))

    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir)
        # Figure out what regions/sources we have results for
        task_result_source_names = [os.path.splitext(f)[0] for f in os.listdir(task_dir)]

        # Go through all datasets there might be results for
        for dataset_filename in os.listdir(datasets_dir):
            dataset_filename = os.path.join(datasets_dir, dataset_filename)
            with open(dataset_filename, 'r') as dataset_file:
                dataset_source_files = [s.strip() for s in dataset_file.readlines()]
                dataset_source_names = [os.path.splitext(os.path.basename(s))[0] for s in dataset_source_files]
                # See if we have all sources for a given dataset
                if set(dataset_source_names).issubset(set(task_result_source_names)):
                    dataset_name = os.path.splitext(os.path.basename(dataset_filename))[0]
                    print(f'Dataset: {dataset_name}')
                    # If so, we can compute some metrics
                    if task_name == 'rgb':
                        total_ssim = 0
                        total_psnr = 0
                        total_mse = 0
                        total_pixels = 0
                        result_images = [os.path.join(task_dir, t) for t in os.listdir(task_dir)]
                        for result_image_file in result_images:
                            result_img = Image.open(result_image_file)

                            # Get the corresponding source .json file from the dataset
                            source_name = os.path.splitext(os.path.basename(result_image_file))[0]
                            candidate_sources = [i for i in dataset_source_files if source_name in i]
                            assert len(candidate_sources) == 1
                            source_json_path = os.path.join(dri_datasets_drive_path, candidate_sources[0])
                            with open(source_json_path, 'r') as source_json_file:
                                source_json = json.load(source_json_file)
                                rgb_label_path = source_json['rgb-label']
                                rgb_label_path = os.path.abspath(os.path.join(os.path.dirname(source_json_path), rgb_label_path))
                                rgb_label_img = Image.open(rgb_label_path)

                            # Mask images based on region's specified mask
                            mask_path = source_json['mask']
                            mask_path = os.path.abspath(os.path.join(os.path.dirname(source_json_path), mask_path))
                            mask_img = Image.open(mask_path)

                            # Generate new blank image and copy masked label image into it
                            blank_img = Image.new('RGB', mask_img.size)
                            blank_img.paste(rgb_label_img, mask=mask_img)
                            rgb_label_img = blank_img.copy()

                            # Do the same with the result image
                            blank_img = Image.new('RGB', mask_img.size)
                            blank_img.paste(result_img, mask=mask_img)
                            result_img = blank_img.copy()

                            # Crop to region's bounding box
                            if source_json.get('bounding-box') is not None:
                                rgb_label_img.crop(source_json['bounding-box'])

                            # Convert images to numpy arrays
                            rgb_label_img = np.array(rgb_label_img)
                            result_img = np.array(result_img)

                            pixels_in_img = rgb_label_img.shape[0] * rgb_label_img.shape[1]
                            total_pixels += pixels_in_img

                            img_ssim = ssim(rgb_label_img, result_img, multichannel=True)
                            total_ssim += img_ssim * pixels_in_img

                            img_mse = mse(rgb_label_img, result_img)
                            total_mse += img_mse * pixels_in_img

                            img_psnr = psnr(rgb_label_img, result_img)
                            total_psnr += img_psnr * pixels_in_img
                            # Use that to find and get the corresponding label image
                        # SSIM
                        print(f'SSIM: {total_ssim / total_pixels}')
                        # MSE
                        print(f'MSE: {total_mse / total_pixels}')
                        # PSNR
                        print(f'PSNR: {total_psnr / pixels_in_img}')
                    elif task_name == 'ink':
                        # Dice
                        # mIOU
                        pass
                    elif task_name == 'volcart-texture':
                        # SSIM
                        # PSNR
                        # MSE
                        pass

    # TODO
    # Do the masking LEFT OFF
    # Do the region bounding if relevant
    # Save them to central leaderboard .csv
    # Save them to results metadata.json
    # Rclone the updated results?


if __name__ == '__main__':
    main()

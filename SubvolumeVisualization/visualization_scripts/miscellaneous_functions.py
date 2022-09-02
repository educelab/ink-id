# miscellaneous_functions.py
#
# script for useful numpy array operations.
#


import numpy as np
from pathlib import Path
from PIL import Image
import re


def stack_to_array(stack_path):
    '''
    Convert subvol stack TIF to numpy
    Args:
        stack_path(str): path/to/directory/containing/numbered/tif/files
    Returns:
        numpy array (3D)
    '''

    dataset = Path(stack_path)

    if not dataset.is_dir():
        print("Error: volume directory not found.")
        return np.zeros((1,1,1))

    else:
        files = list(dataset.glob('*.tif'))
        files.sort(key=lambda f: int(re.sub(r'[^0-9]*', "", str(f))))
        vol_array = [] 

        for f in files:
          vol_array.append(np.array(Image.open(f), dtype=np.float32))
        
        # convert to numpy
        vol_array = np.array(vol_array)
        
        # sanity check
        print("numpy array shape: ", np.shape(vol_array))

            
        return vol_array


def save_array_as_npy(n_array, outfile):
    '''
    Save numpy array as an .npy file at a specified path
    Args:
        n_array(numpy array)
        outfile(str): paty/to/output/file(.npy)
    Returns:
        None
    '''
    if outfile[-4:] != '.npy':
        print("outfile must have an .npy extension")

    with open(outfile, 'wb') as f:
        np.save(f, n_array)


def numpy_binary_to_array(npy_file):
    '''
    Load .npy file
    Args:
        npy_file(str): path/to/npy/file
    Returns:
        numpy array (3D)
    '''

    datafile = Path(npy_file)
    
    if not datafile.suffix == '.npy':
        print("Error: numpy binary file not found.")
        return np.zeros((1,1,1))

    else:
        with open(npy_file, 'rb') as f:
            array = np.load(f)

    return array



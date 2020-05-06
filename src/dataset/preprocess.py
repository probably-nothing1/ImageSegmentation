import os
import numpy as np


def preprocess_masks(folderpath, input_filepath, out_filepath):
    input_filepath = os.path.join(folderpath, input_filepath)
    out_filepath = os.path.join(folderpath, out_filepath)

    array = np.load(input_filepath)
    array = np.rollaxis(array, 3, 1)
    array = array.astype('float32')
    array[array>0.0] = 1.0
    np.save(out_filepath, array)

def preprocess_images(folderpath, input_filepath, out_filepath):
    input_filepath = os.path.join(folderpath, input_filepath)
    out_filepath = os.path.join(folderpath, out_filepath)

    array = np.load(input_filepath)
    array = np.rollaxis(array, 3, 1)
    array = array.astype('float32')
    np.save(out_filepath, array)

def preprocess_data(folderpath):
    preprocess_images(folderpath, 'gsn_img_uint8.npy', 'train_img_preprocessed.npy')
    preprocess_masks(folderpath, 'gsn_msk_uint8.npy', 'train_mask_preprocessed.npy')
    preprocess_images(folderpath, 'test_gsn_image.npy', 'test_img_preprocessed.npy')
    preprocess_masks(folderpath, 'test_gsn_mask.npy', 'test_mask_preprocessed.npy')
import os

def preprocess_np_array(folderpath, input_filepath, out_filepath):
    input_filepath = os.path.join(folderpath, input_filepath)
    out_filepath = os.path.join(folderpath, out_filepath)

    train_images = np.load(input_filepath)
    train_images = np.rollaxis(train_images, 3, 1)
    train_images = train_images.astype('float32')
    np.save(out_filepath, train_images)

def preprocess_data(folderpath):
    preprocess_np_array(folderpath, 'gsn_img_uint8.npy', 'train_img_preprocessed.npy')
    preprocess_np_array(folderpath, 'gsn_msk_uint8.npy', 'train_mask_preprocessed.npy')
    preprocess_np_array(folderpath, 'test_gsn_image.npy', 'test_img_preprocessed.npy')
    preprocess_np_array(folderpath, 'test_gsn_mask.npy', 'test_mask_preprocessed.npy')
import os
import pickle
from scipy import ndimage
import numpy as np

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    # get list of files in folder
    image_files = os.listdir(folder)
    # init np's n-dimensional Array
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            # feature scaling
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def may_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


train_datasets = may_pickle(['C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_large/notMNIST_large/A/',
                             'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_large/notMNIST_large/B/',
                             'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_large/notMNIST_large/C/',
                             'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_large/notMNIST_large/D/',
                             'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_large/notMNIST_large/E/'],
                            45000)
test_datasets = may_pickle(['C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_small/notMNIST_small/A/',
                            'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_small/notMNIST_small/B/',
                            'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_small/notMNIST_small/C/',
                            'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_small/notMNIST_small/D/',
                            'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_small/notMNIST_small/E/'],
                           1800)

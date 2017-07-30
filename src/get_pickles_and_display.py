import pickle
import numpy as np
import matplotlib.pyplot as plt

all_images = []
image_size = 28
from collections import Counter


def get_pickle_and_plot(folders, plots=2):
    for folder in folders:
        image_array = np.array([0])
        try:
            with open(folder + ".pickle", 'rb') as f:
                image_array = pickle.load(f)
        except Exception as e:
            print("issue opening the file/reading pickle" + e)
        if image_array.shape[0] > plots:
            image_array = image_array[:plots]
            all_images.append(image_array)
    fig = plt.figure(1)
    index = 1
    for i in all_images:
        for image in i:
            fig.add_subplot(len(all_images), len(i), index)
            plt.imshow(image)
            index += 1
    plt.show()


"""get_pickle_and_plot(['C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_large/notMNIST_large/A/',
                             'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_large/notMNIST_large/B/',
                             'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_large/notMNIST_large/C/',
                             'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_large/notMNIST_large/D/',
                             'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_large/notMNIST_large/E/'],
                            2)"""


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        pickle_file = pickle_file + '.pickle'
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


train_size = 100000
valid_size = 5000
test_size = 5000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    ['C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_large/notMNIST_large/A/',
     'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_large/notMNIST_large/B/',
     'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_large/notMNIST_large/C/',
     'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_large/notMNIST_large/D/',
     'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_large/notMNIST_large/E/'], train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(
    ['C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_small/notMNIST_small/A/',
     'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_small/notMNIST_small/B/',
     'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_small/notMNIST_small/C/',
     'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_small/notMNIST_small/D/',
     'C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/notMNIST_small/notMNIST_small/E/'], test_size)


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
print(Counter(train_labels), Counter(test_labels))


def validate_data(dataset, labels, plot=False):
    print("mean and deviation(should be close to zero): ", np.mean(dataset), np.std(dataset))

    unique, counts = np.unique(labels, return_counts=True)
    a = dict(zip(unique, counts))
    print("Count(equally distributed amongst labels): ", a)
    if plot:
        plt.bar(range(len(a)), a.values(), align='center')
        plt.xticks(range(len(a)), a.keys())

        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title('Balanced data')
        plt.legend()

        plt.tight_layout()
        plt.show()


def save_final(tr_dataset, tr_labels, cv_dataset, cv_labels, testing_dataset, testing_labels):
    try:
        with open("C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/AtoE.pickle", "wb") as f:
            save = {
                'train_dataset': tr_dataset,
                'train_labels': tr_labels,
                'valid_dataset': cv_dataset,
                'valid_labels': cv_labels,
                'test_dataset': testing_dataset,
                'test_labels': testing_labels
            }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Error saving file', e)
        raise


print('Training:', train_dataset.shape, train_labels.shape)
validate_data(train_dataset, train_labels)  # Add true to plot

print('Validation:', valid_dataset.shape, valid_labels.shape)
validate_data(valid_dataset, valid_labels)  # Add true to plot

print('Testing:', test_dataset.shape, test_labels.shape)
validate_data(test_dataset, test_labels)  # Add true to plot

save_final(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)

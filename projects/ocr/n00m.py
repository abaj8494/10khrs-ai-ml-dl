import numpy as np
import math

DATA_DIR = 'data/'
DATASET = 'mnist'  # or 'fmnist'
TEST_DATA_FILENAME = DATA_DIR + DATASET + '/t10k-images-idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + DATASET + '/t10k-labels-idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + DATASET + '/train-images-idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + DATASET + '/train-labels-idx1-ubyte'


def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')


def read_images(filename, n_max_images=None):
    """Reads IDX image file and returns a list of images (2D lists of 1-byte pixels)."""
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_cols = bytes_to_int(f.read(4))
        for _ in range(n_images):
            image = []
            for _ in range(n_rows):
                row = []
                for _ in range(n_cols):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images


def read_labels(filename, n_max_labels=None):
    """Reads IDX label file and returns a list of integer labels."""
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for _ in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return labels


def flatten_list(image_2d):
    """Convert a 2D list of single-byte objects into a 1D list."""
    return [pix[0] for row in image_2d for pix in row]


def get_garment_from_label(label):
    return [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot',
    ][label]


def vectorized_knn(X_train, y_train, X_test, k=7):
    """
    Vectorized kNN using NumPy.
    - X_train: shape (n_train, 784)
    - X_test: shape (n_test, 784)
    - Returns y_pred: shape (n_test,)
    """
    # 1) Compute the distance-squared matrix of shape (n_test, n_train)
    #    We can skip sqrt for the nearest-neighbor check (sqrt is monotonic).
    #    distances[i, j] = sum( (X_test[i] - X_train[j])^2 )
    # Using broadcasting: X_test -> (n_test, 1, 784), X_train -> (1, n_train, 784)
    
    # Convert to float32 or float64. If data is uint8, the difference can remain int,
    # but let's ensure we don't overflow by upcasting:
    X_train_f = X_train.astype(np.float32)
    X_test_f = X_test.astype(np.float32)
    
    # shape (n_test, n_train, 784):
    diff = X_test_f[:, np.newaxis, :] - X_train_f[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)  # shape (n_test, n_train)
    
    n_test = X_test.shape[0]
    y_pred = np.empty(n_test, dtype=y_train.dtype)
    
    # 2) For each test sample, find the k nearest training samples
    #    We'll use np.argpartition for partial sort to get the smallest k distances
    for i in range(n_test):
        # Indices of the k smallest distances
        neighbors_idx = np.argpartition(dist_sq[i], k)[:k]
        # labels of those neighbors
        neighbor_labels = y_train[neighbors_idx]
        # majority vote
        vals, counts = np.unique(neighbor_labels, return_counts=True)
        y_pred[i] = vals[np.argmax(counts)]
        
        if i % 1000 == 0:
            print(f"Processed test sample {i}/{n_test}")
    
    return y_pred


def main():
    n_train = 60000
    n_test = 10000
    k = 7

    print(f"Dataset: {DATASET}")
    print(f"n_train: {n_train}")
    print(f"n_test: {n_test}")
    print(f"k: {k}")

    # 1) Read the data from IDX files
    #    (Make sure you have data/mnist/ or data/fmnist/ with the IDX files)
    raw_train = read_images(TRAIN_DATA_FILENAME, n_train)
    y_train = np.array(read_labels(TRAIN_LABELS_FILENAME, n_train), dtype=np.int32)
    raw_test = read_images(TEST_DATA_FILENAME, n_test)
    y_test = np.array(read_labels(TEST_LABELS_FILENAME, n_test), dtype=np.int32)
    
    # 2) Flatten each image from shape (28,28) to (784,)
    #    Then convert to a NumPy array
    #    shape -> (n_train, 784)
    X_train = np.array([flatten_list(img) for img in raw_train], dtype=np.dtype('B'))
    X_test = np.array([flatten_list(img) for img in raw_test], dtype=np.dtype('B'))
    
    print("Finished loading data. Shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    # 3) Run vectorized KNN (without parallelization)
    y_pred = vectorized_knn(X_train, y_train, X_test, k)

    # 4) Compute accuracy
    accuracy = np.mean(y_pred == y_test)
    
    # 5) Print results
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    if DATASET == 'fmnist':
        # Optionally see how the first 10 predictions map to garment names
        garments_pred = [get_garment_from_label(label) for label in y_pred[:10]]
        print(f"First 10 predicted garments: {garments_pred}")
    else:
        print(f"First 10 predicted digits: {y_pred[:10]}")


if __name__ == '__main__':
    main()


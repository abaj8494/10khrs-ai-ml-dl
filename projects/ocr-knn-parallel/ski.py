import numpy as np
from sklearn.neighbors import KNeighborsClassifier

DATA_DIR = 'data/'
DATASET = 'mnist'  # or 'fmnist'
TEST_DATA_FILENAME = DATA_DIR + DATASET + '/t10k-images-idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + DATASET + '/t10k-labels-idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + DATASET + '/train-images-idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + DATASET + '/train-labels-idx1-ubyte'

def bytes_to_int(b):
    return int.from_bytes(b, 'big')

def read_images(filename, n_max_images=None):
    with open(filename, 'rb') as f:
        _ = f.read(4)
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_cols = bytes_to_int(f.read(4))
        
        images = []
        for _ in range(n_images):
            img = [f.read(1) for _ in range(n_rows*n_cols)]
            images.append(img)
    return images

def read_labels(filename, n_max_labels=None):
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        labels = []
        for _ in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return labels

def main():
    n_train = 60000
    n_test = 10000
    k = 7

    print(f"Dataset: {DATASET}")
    print(f"n_train: {n_train}")
    print(f"n_test: {n_test}")
    print(f"k: {k}")

    # 1) Read data
    raw_train = read_images(TRAIN_DATA_FILENAME, n_train)
    y_train = np.array(read_labels(TRAIN_LABELS_FILENAME, n_train), dtype=np.int32)
    raw_test = read_images(TEST_DATA_FILENAME, n_test)
    y_test = np.array(read_labels(TEST_LABELS_FILENAME, n_test), dtype=np.int32)

    # 2) Convert each image to shape (784,) as uint8
    X_train = np.zeros((n_train, 784), dtype=np.uint8)
    for i, img in enumerate(raw_train):
        X_train[i] = [p[0] for p in img]
    del raw_train

    X_test = np.zeros((n_test, 784), dtype=np.uint8)
    for i, img in enumerate(raw_test):
        X_test[i] = [p[0] for p in img]
    del raw_test

    print("Shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    # 3) Use scikit-learn’s KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, algorithm="auto")
    knn.fit(X_train, y_train)

    # 4) Predict on test set
    y_pred = knn.predict(X_test)

    # 5) Evaluate accuracy
    accuracy = (y_pred == y_test).mean() * 100
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()


import os
import math
import multiprocessing
from multiprocessing import Process, Queue
import concurrent.futures
import threading

##################################################
# Same data loading functions as your snippet
##################################################

DATA_DIR = 'data/'
DATASET = 'fmnist'  # or 'mnist' if you prefer
TEST_DATA_FILENAME = DATA_DIR + DATASET + '/t10k-images-idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + DATASET + '/t10k-labels-idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + DATASET + '/train-images-idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + DATASET + '/train-labels-idx1-ubyte'


def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')

def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_cols = bytes_to_int(f.read(4))
        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_cols):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images

def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return labels

def flatten_list(subl):
    return [pixel for sublist in subl for pixel in sublist]

def extract_features(X):
    """ Convert 2D list-of-bytes into a 1D list-of-bytes for each image. """
    return [flatten_list(sample) for sample in X]

def bytes_to_int_fast(b):
    """Slightly faster local version to avoid re-calling int.from_bytes in a loop."""
    return b[0]  # since each pixel is a single byte

##################################################
# Distance + kNN classification
##################################################

def dist(x, y):
    """
    Euclidean distance. x, y are lists of single-byte values.
    Each element is a 'bytes' object of length 1, e.g. b'\\x7f'.
    """
    # For speed, we can interpret b[0] directly:
    # (Note that for n_train=60k, n_test=10k, this is STILL huge in Python!)
    s = 0
    for x_i, y_i in zip(x, y):
        diff = (x_i[0] - y_i[0])
        s += diff * diff
    return math.sqrt(s)

def classify_one_sample(test_sample_idx, test_sample, X_train, y_train, k):
    """
    Compute kNN for a single test sample.
    We'll print the test_sample_idx to show progress.
    """
    print(f"PID {os.getpid()} - Thread {threading.current_thread().name} "
          f"is processing test sample #{test_sample_idx}")

    # Compute distance to all training samples
    training_distances = [dist(train_sample, test_sample) for train_sample in X_train]
    
    # Find k nearest
    sorted_idxs = sorted(range(len(training_distances)), key=lambda i: training_distances[i])[:k]
    neighbors = [y_train[i] for i in sorted_idxs]
    # majority vote
    return max(neighbors, key=neighbors.count)

##################################################
# Process-level worker
# Receives a CHUNK of X_test (and corresponding chunk indices)
# Then spawns threads for each test sample in that chunk
##################################################

def process_chunk(proc_id, start_idx, X_test_chunk, X_train, y_train, k, result_queue):
    """
    This function runs in a separate process.
    It uses a thread pool to handle each test sample in X_test_chunk.
    """
    pid = os.getpid()
    print(f"Process {proc_id} started (PID={pid}), handling {len(X_test_chunk)} test samples.")

    # We'll store predictions in local list
    local_preds = [None] * len(X_test_chunk)

    # Let's define how many threads within this process
    num_threads = 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {}
        for i, test_sample in enumerate(X_test_chunk):
            global_test_index = start_idx + i  # The absolute index in the full test set
            future = executor.submit(
                classify_one_sample,
                global_test_index,
                test_sample,
                X_train,
                y_train,
                k
            )
            futures[future] = i  # so we know which local index it corresponds to

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            local_i = futures[future]
            pred_label = future.result()
            local_preds[local_i] = pred_label

    # Put back to main via a queue: (proc_id, local_preds)
    result_queue.put((proc_id, local_preds))
    print(f"Process {proc_id} finished (PID={pid}).")

##################################################
# Top-level function that spawns 24 processes
##################################################

def knn_parallel(X_train, y_train, X_test, k=7, num_processes=24):
    """
    We split the test set into <num_processes> chunks.
    Each chunk is handled by a separate process, which in turn
    spawns threads to classify its chunk of test samples.
    """
    total_test = len(X_test)
    chunk_size = math.ceil(total_test / num_processes)

    result_queue = multiprocessing.Queue()
    processes = []

    for proc_id in range(num_processes):
        start_idx = proc_id * chunk_size
        if start_idx >= total_test:
            break
        end_idx = min(start_idx + chunk_size, total_test)
        X_test_chunk = X_test[start_idx:end_idx]
        
        p = Process(
            target=process_chunk,
            args=(proc_id, start_idx, X_test_chunk, X_train, y_train, k, result_queue)
        )
        p.start()
        processes.append(p)

    # Collect partial results from the queue
    results_by_proc = [None]*num_processes
    for _ in processes:
        proc_id, preds = result_queue.get()
        results_by_proc[proc_id] = preds

    # Wait for all processes
    for p in processes:
        p.join()

    # Flatten predictions in the correct order
    all_predictions = []
    for chunk_preds in results_by_proc:
        if chunk_preds is not None:
            all_predictions.extend(chunk_preds)

    return all_predictions

##################################################
# Put it all together in main()
##################################################

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

def main():
    # FULL dataset sizes
    n_train = 60000
    n_test = 10000
    k = 7
    print(f"Dataset: {DATASET}")
    print(f"n_train: {n_train}")
    print(f"n_test: {n_test}")
    print(f"k: {k}")

    # 1) Read data
    X_train = read_images(TRAIN_DATA_FILENAME, n_train)
    y_train = read_labels(TRAIN_LABELS_FILENAME, n_train)
    X_test = read_images(TEST_DATA_FILENAME, n_test)
    y_test = read_labels(TEST_LABELS_FILENAME, n_test)

    # 2) Flatten images into 1D lists-of-bytes
    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    # 3) Run parallel kNN
    y_pred = knn_parallel(X_train, y_train, X_test, k=k, num_processes=24)

    # 4) Compute accuracy
    correct = sum(int(p == t) for p, t in zip(y_pred, y_test))
    accuracy = correct / len(y_test)

    # 5) Print results
    if DATASET == 'fmnist':
        garments_pred = [get_garment_from_label(label) for label in y_pred]
        print(f"Predicted garments (first 50 shown): {garments_pred[:50]}")
    else:
        print(f"Predicted labels (first 50 shown): {y_pred[:50]}")

    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()

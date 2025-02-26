from sklearn.datasets import fetch_openml
import numpy as np

def load_data():
    mnist_data = fetch_openml("mnist_784", version=1)
    X = mnist_data.data.to_numpy(dtype=np.float32)  # Convert to NumPy array
    y = mnist_data.target.astype(int)  # Convert labels to integers
    X /= 255.0  # Normalize to range [0, 1]
    y_onehot = np.eye(10)[y]  # One-hot encode the labels

    # Split into training and test sets
    split = 60000
    return X[:split], X[split:], y_onehot[:split], y_onehot[split:]

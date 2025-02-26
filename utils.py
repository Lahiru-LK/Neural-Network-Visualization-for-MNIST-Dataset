import numpy as np

# Function to calculate cross-entropy loss
def calculate_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))  # Add small epsilon to avoid log(0)

# Function to count active neurons in a layer
def count_active_neurons(layer_output):
    return np.sum(layer_output > 0, axis=1).mean()  # Average number of active neurons per sample in the batch

# Function to calculate weight statistics
def calculate_weight_stats(weights):
    mean = np.mean(np.abs(weights))
    std = np.std(weights)
    return mean, std

# Function to calculate accuracy for a specific digit
def calculate_digit_accuracy(y_true, y_pred, digit):
    true_labels = np.argmax(y_true, axis=1)  # Convert one-hot encoding to digit labels
    pred_labels = np.argmax(y_pred, axis=1)  # Convert predictions to digit labels

    # Filter for the specific digit
    digit_indices = (true_labels == digit)
    if np.sum(digit_indices) == 0:
        return 0.0  # Avoid division by zero if no samples for the digit

    digit_accuracy = np.mean(pred_labels[digit_indices] == true_labels[digit_indices])
    return digit_accuracy

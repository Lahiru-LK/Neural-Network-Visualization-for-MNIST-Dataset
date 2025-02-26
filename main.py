from data_loader import load_data
from neural_network import NeuralNetwork
from visualization import train_nn_sparse_with_digit_accuracy

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
INPUT_SIZE = 784
HIDDEN_SIZE_1 = 1024
HIDDEN_SIZE_2 = 512
DROPOUT_RATE = 0.2
OUTPUT_SIZE = 10


def main():
    x_train, x_test, y_train, y_test = load_data()
    nn = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE)
    train_nn_sparse_with_digit_accuracy(nn, x_train, y_train, x_test, y_test,
                                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                                        learning_rate=LEARNING_RATE)

if __name__ == "__main__":
    main()

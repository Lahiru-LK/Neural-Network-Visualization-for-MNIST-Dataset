import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        self.weights_input_hidden1 = np.random.randn(input_size, hidden_size_1) * np.sqrt(3 / input_size)
        self.weights_hidden1_hidden2 = np.random.randn(hidden_size_1, hidden_size_2) * np.sqrt(2 / hidden_size_1)
        self.weights_hidden2_output = np.random.randn(hidden_size_2, output_size) * np.sqrt(1 / hidden_size_2)
        self.bias_hidden1 = np.zeros((1, hidden_size_1))
        self.bias_hidden2 = np.zeros((1, hidden_size_2))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, x):
        self.hidden1_input = np.dot(x, self.weights_input_hidden1) + self.bias_hidden1
        self.hidden1_output = np.maximum(0, self.hidden1_input)  # ReLU Activation
        self.hidden2_input = np.dot(self.hidden1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
        self.hidden2_output = np.maximum(0, self.hidden2_input)  # ReLU Activation
        self.output_input = np.dot(self.hidden2_output, self.weights_hidden2_output) + self.bias_output
        return self.softmax(self.output_input)

    def backward(self, x, y, output, learning_rate):
        output_error = output - y
        hidden2_error = np.dot(output_error, self.weights_hidden2_output.T) * (self.hidden2_input > 0)
        hidden1_error = np.dot(hidden2_error, self.weights_hidden1_hidden2.T) * (self.hidden1_input > 0)

        self.weights_hidden2_output -= learning_rate * np.dot(self.hidden2_output.T, output_error)
        self.bias_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        self.weights_hidden1_hidden2 -= learning_rate * np.dot(self.hidden1_output.T, hidden2_error)
        self.bias_hidden2 -= learning_rate * np.sum(hidden2_error, axis=0, keepdims=True)
        self.weights_input_hidden1 -= learning_rate * np.dot(x.T, hidden1_error)
        self.bias_hidden1 -= learning_rate * np.sum(hidden1_error, axis=0, keepdims=True)

    @staticmethod
    def softmax(x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

import numpy as np
import random

# A simple, two-layer neural network using NumPy.
# This program simulates the core logic of the provided HTML visualization,
# demonstrating how a network learns through backpropagation and gradient descent.

class NeuralNetwork:
    """
    A simple neural network with one hidden layer.
    It supports Sigmoid, ReLU, and Tanh activation functions.
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        """
        Initializes the neural network with random weights and biases.

        Args:
            input_size (int): Number of input neurons.
            hidden_size (int): Number of hidden neurons.
            output_size (int): Number of output neurons.
            learning_rate (float): The step size for gradient descent.
        """
        self.learning_rate = learning_rate

        # Initialize weights and biases for the hidden layer
        self.weights_h = np.random.randn(input_size, hidden_size) * 0.1
        self.biases_h = np.zeros((1, hidden_size))

        # Initialize weights and biases for the output layer
        self.weights_o = np.random.randn(hidden_size, output_size) * 0.1
        self.biases_o = np.zeros((1, output_size))

        # Store activation sums and outputs for the backward pass
        self.activation_sums = {}
        self.outputs = {}

    def _sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        """Derivative of the sigmoid function."""
        return x * (1 - x)

    def _relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        """Derivative of the ReLU function."""
        return np.where(x > 0, 1, 0)

    def _tanh(self, x):
        """Tanh activation function."""
        return np.tanh(x)

    def _tanh_derivative(self, x):
        """Derivative of the Tanh function."""
        return 1.0 - np.tanh(x)**2

    def forward_pass(self, inputs, activation_func='sigmoid'):
        """
        Performs the forward pass to calculate the network's output.

        Args:
            inputs (numpy.array): The input data.
            activation_func (str): The activation function to use in the hidden layer.

        Returns:
            numpy.array: The final predicted output.
        """
        # Determine the activation function and its derivative
        if activation_func == 'sigmoid':
            self.activation = self._sigmoid
            self.activation_derivative = self._sigmoid_derivative
        elif activation_func == 'relu':
            self.activation = self._relu
            self.activation_derivative = self._relu_derivative
        elif activation_func == 'tanh':
            self.activation = self._tanh
            self.activation_derivative = self._tanh_derivative
        else:
            raise ValueError("Invalid activation function specified.")

        # Hidden layer calculations
        self.activation_sums['h'] = np.dot(inputs, self.weights_h) + self.biases_h
        self.outputs['h'] = self.activation(self.activation_sums['h'])

        # Output layer calculations
        self.activation_sums['o'] = np.dot(self.outputs['h'], self.weights_o) + self.biases_o
        self.outputs['o'] = self._sigmoid(self.activation_sums['o'])

        return self.outputs['o']

    def backward_pass(self, inputs, targets):
        """
        Performs the backward pass (backpropagation) to calculate gradients.

        Args:
            inputs (numpy.array): The input data.
            targets (numpy.array): The true target values.

        Returns:
            tuple: Gradients for weights and biases of both layers.
        """
        # Calculate the error for the output layer
        output_error = targets - self.outputs['o']
        
        # Calculate the delta for the output layer
        output_delta = output_error * self._sigmoid_derivative(self.outputs['o'])
        
        # Calculate the error for the hidden layer
        hidden_error = np.dot(output_delta, self.weights_o.T)
        
        # Calculate the delta for the hidden layer
        hidden_delta = hidden_error * self.activation_derivative(self.outputs['h'])

        # Calculate gradients for the output layer
        d_weights_o = np.dot(self.outputs['h'].T, output_delta)
        d_biases_o = np.sum(output_delta, axis=0, keepdims=True)

        # Calculate gradients for the hidden layer
        d_weights_h = np.dot(inputs.T, hidden_delta)
        d_biases_h = np.sum(hidden_delta, axis=0, keepdims=True)

        return d_weights_h, d_biases_h, d_weights_o, d_biases_o

    def update_weights(self, d_weights_h, d_biases_h, d_weights_o, d_biases_o):
        """
        Updates the weights and biases using gradient descent.
        """
        self.weights_o += self.learning_rate * d_weights_o
        self.biases_o += self.learning_rate * d_biases_o
        self.weights_h += self.learning_rate * d_weights_h
        self.biases_h += self.learning_rate * d_biases_h

    def train(self, dataset, epochs, activation_func='sigmoid'):
        """
        Trains the network for a specified number of epochs.

        Args:
            dataset (list): List of dictionaries with input and target data.
            epochs (int): Number of training iterations.
            activation_func (str): The activation function for the hidden layer.
        """
        for epoch in range(epochs):
            total_error = 0
            for data in dataset:
                inputs = np.array([[data['study'] / 10, data['sleep'] / 10]])
                targets = np.array([[data['target']]])

                # Forward pass
                prediction = self.forward_pass(inputs, activation_func)

                # Calculate mean squared error (MSE)
                error = np.mean((targets - prediction) ** 2)
                total_error += error

                # Backward pass
                d_weights_h, d_biases_h, d_weights_o, d_biases_o = self.backward_pass(inputs, targets)

                # Update weights
                self.update_weights(d_weights_h, d_biases_h, d_weights_o, d_biases_o)

            avg_error = total_error / len(dataset)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Average Error: {avg_error:.6f}")

        print("\nTraining complete.")
        print("Final Weights and Biases:")
        print("Hidden Weights:\n", self.weights_h)
        print("Hidden Biases:\n", self.biases_h)
        print("Output Weights:\n", self.weights_o)
        print("Output Biases:\n", self.biases_o)
        
        # Test the network on a single data point after training
        test_input = dataset[0]
        test_prediction = self.forward_pass(np.array([[test_input['study'] / 10, test_input['sleep'] / 10]]))
        print("\n--- Final Test ---")
        print(f"Input: Study={test_input['study']} hrs, Sleep={test_input['sleep']} hrs")
        print(f"Actual Target: {'Pass' if test_input['target'] == 1 else 'Fail'} ({test_input['target']})")
        print(f"Predicted Probability: {test_prediction[0][0] * 100:.2f}%")

def generate_dataset(size=100):
    """
    Generates a synthetic dataset for demonstration purposes.
    The target (Pass/Fail) is based on a noisy linear combination of study and sleep hours.
    """
    dataset = []
    for _ in range(size):
        study = round(random.uniform(1, 10), 1)
        sleep = round(random.uniform(1, 10), 1)
        # Create a simple, noisy linear boundary
        target = 1 if (study + sleep) / 20 + random.uniform(-0.1, 0.1) > 0.5 else 0
        dataset.append({'study': study, 'sleep': sleep, 'target': target})
    return dataset

if __name__ == "__main__":
    # Parameters
    INPUT_SIZE = 2
    HIDDEN_SIZE = 2
    OUTPUT_SIZE = 1
    LEARNING_RATE = 0.5
    EPOCHS = 200

    # 1. Generate the dataset
    training_data = generate_dataset(200)

    # 2. Instantiate the neural network
    nn = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE)

    # 3. Train the network
    print("Starting training...")
    nn.train(training_data, EPOCHS, activation_func='tanh')

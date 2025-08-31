import numpy as np

def sigmoid(x):
    """Sigmoid activation function. Maps any value to a value between 0 and 1."""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU (Rectified Linear Unit) activation function. Returns x if x > 0, otherwise 0."""
    return np.maximum(0, x)

def tanh(x):
    """Tanh (Hyperbolic Tangent) activation function. Maps any value to a value between -1 and 1."""
    return np.tanh(x)

class SingleNeuron:
    """A simple class representing a single neuron."""

    def __init__(self, weights, bias, activation_function):
        # Initialize the neuron with given weights, bias, and activation function.
        self.weights = np.array(weights)
        self.bias = bias
        
        # Select the activation function based on the string input.
        if activation_function == 'sigmoid':
            self.activation_func = sigmoid
        elif activation_function == 'relu':
            self.activation_func = relu
        elif activation_function == 'tanh':
            self.activation_func = tanh
        else:
            raise ValueError("Invalid activation function specified. Choose 'sigmoid', 'relu', or 'tanh'.")

    def forward_propagation(self, inputs):
        """
        Calculates the neuron's output for a given set of inputs.

        Args:
            inputs (list or np.array): A list of input values (e.g., [study_hours, sleep_hours]).

        Returns:
            float: The neuron's output after applying the activation function.
        """
        # 1. Calculate the weighted sum of inputs and add the bias.
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        
        # 2. Pass the result through the activation function to get the final output.
        output = self.activation_func(weighted_sum)
        
        return output

# --- Main Program ---

if __name__ == "__main__":
    # Define inputs, weights, and bias
    x1_study_hours = 5.0
    x2_sleep_hours = 7.0
    
    w1_study_weight = 1.5
    w2_sleep_weight = -0.8
    
    bias = 0.2
    
    inputs = [x1_study_hours, x2_sleep_hours]
    weights = [w1_study_weight, w2_sleep_weight]
    
    # Create the neuron with the Sigmoid activation function
    my_neuron = SingleNeuron(weights, bias, activation_function='sigmoid')

    # Run the forward propagation to get the output
    pass_probability = my_neuron.forward_propagation(inputs)

    # Display the results
    print("Inputs:")
    print(f"  Study Hours (x1): {x1_study_hours}")
    print(f"  Sleep Hours (x2): {x2_sleep_hours}\n")

    print("Weights & Bias:")
    print(f"  Weight w1 (Study): {w1_study_weight}")
    print(f"  Weight w2 (Sleep): {w2_sleep_weight}")
    print(f"  Bias: {bias}\n")

    print("Calculation:")
    print(f"  Weighted Sum (z) = (x1 * w1) + (x2 * w2) + bias")
    print(f"  z = ({x1_study_hours} * {w1_study_weight}) + ({x2_sleep_hours} * {w2_sleep_weight}) + {bias}")
    print(f"  z = {(x1_study_hours * w1_study_weight) + (x2_sleep_hours * w2_sleep_weight) + bias:.2f}")
    
    print(f"  Output = Sigmoid(z)")
    print(f"  Output = {pass_probability:.4f}\n")

    print(f"Result: The probability of passing is {pass_probability * 100:.2f}%")
    
    # --- Example with a different activation function ---
    print("\n" + "="*50 + "\n")
    print("Example with ReLU activation function:")
    relu_neuron = SingleNeuron(weights, bias, activation_function='relu')
    relu_output = relu_neuron.forward_propagation(inputs)
    print(f"Output using ReLU: {relu_output:.2f}")

    # --- Another example with different inputs and weights ---
    print("\n" + "="*50 + "\n")
    print("New example with low study hours and high sleep hours:")
    new_inputs = [2.0, 9.0]
    new_weights = [0.8, -0.1]
    
    new_neuron = SingleNeuron(new_weights, bias=0.1, activation_function='sigmoid')
    new_probability = new_neuron.forward_propagation(new_inputs)
    
    print(f"  Study Hours: {new_inputs[0]}")
    print(f"  Sleep Hours: {new_inputs[1]}")
    print(f"  New Weights: w1={new_weights[0]}, w2={new_weights[1]}")
    print(f"  New Bias: {0.1}")
    print(f"Resulting Pass Probability: {new_probability * 100:.2f}%")

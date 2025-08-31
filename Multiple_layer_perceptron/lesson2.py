import numpy as np

# Define the activation functions
def sigmoid(x):
    """Sigmoid activation function. Maps any value to a range between 0 and 1."""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU (Rectified Linear Unit) activation function. Outputs the input if positive, otherwise 0."""
    return np.maximum(0, x)

def tanh(x):
    """Tanh (Hyperbolic Tangent) activation function. Maps any value to a range between -1 and 1."""
    return np.tanh(x)

def forward_propagation(study_hours, sleep_hours, weights, biases, activation_func_name='sigmoid'):
    """
    Performs the forward pass of a simple MLP.
    
    Args:
        study_hours (float): The number of study hours.
        sleep_hours (float): The number of sleep hours.
        weights (dict): A dictionary containing the weight values.
        biases (dict): A dictionary containing the bias values.
        activation_func_name (str): The name of the activation function to use for the hidden layer.
    
    Returns:
        float: The final probability of passing the exam.
    """
    
    # Select the activation function based on the input string
    activation_funcs = {
        'sigmoid': sigmoid,
        'relu': relu,
        'tanh': tanh
    }
    hidden_activation = activation_funcs.get(activation_func_name, sigmoid) # Default to sigmoid
    
    # 1. Normalize the inputs to a 0-1 range for better network performance
    study_normalized = study_hours / 10.0
    sleep_normalized = sleep_hours / 10.0
    
    # 2. Hidden Layer Calculation
    z1 = (study_normalized * weights['w11']) + (sleep_normalized * weights['w21']) + biases['b1']
    h1_output = hidden_activation(z1)
    
    z2 = (study_normalized * weights['w12']) + (sleep_normalized * weights['w22']) + biases['b2']
    h2_output = hidden_activation(z2)
    
    # 3. Output Layer Calculation
    zo = (h1_output * weights['wo1']) + (h2_output * weights['wo2']) + biases['bo']
    output_probability = sigmoid(zo) # Use sigmoid for the final probability
    
    return output_probability

# --- SET UP THE NETWORK PARAMETERS ---
# These values correspond to the default values in the web visualization.
weights = {
    'w11': 1.0, 'w21': -0.5,
    'w12': -1.0, 'w22': 1.5,
    'wo1': 2.0, 'wo2': 1.0
}

biases = {
    'b1': 0.5, 'b2': -0.2, 'bo': 0.0
}

# --- EXAMPLE USAGE ---
study_input = 1.0
sleep_input = 10.0

print(f"--- Inputs: Study Hours = {study_input}, Sleep Hours = {sleep_input} ---")

# Example 1: Using Sigmoid activation
prob_sigmoid = forward_propagation(study_input, sleep_input, weights, biases, 'sigmoid')
print(f"Using Sigmoid: Probability of Passing = {prob_sigmoid:.4f} ({prob_sigmoid * 100:.2f}%)")

# Example 2: Using ReLU activation
prob_relu = forward_propagation(study_input, sleep_input, weights, biases, 'relu')
print(f"Using ReLU: Probability of Passing = {prob_relu:.4f} ({prob_relu * 100:.2f}%)")

# Example 3: Using Tanh activation
prob_tanh = forward_propagation(study_input, sleep_input, weights, biases, 'tanh')
print(f"Using Tanh: Probability of Passing = {prob_tanh:.4f} ({prob_tanh * 100:.2f}%)")

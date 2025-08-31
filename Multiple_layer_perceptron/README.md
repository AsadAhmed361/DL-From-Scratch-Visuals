# MLP Visualization and Code

This project provides an interactive web visualization of a simple **Multi-Layer Perceptron (MLP)** and a corresponding Python script that implements the core logic. It's designed to help you understand the fundamental concepts of a neural network's forward propagation in a hands-on way.

---

<img width="1716" height="544" alt="d" src="https://github.com/user-attachments/assets/13e645fa-3eb6-463e-801f-29a7312726a0" />


## Web Visualization (`lesson2.html`)

The interactive web application lets you manipulate the network's **inputs, weights, and biases** in real-time to see how they affect the final output. It's a great tool for building an intuitive understanding of how neural networks make predictions.

### Features

- **Interactive Controls:**  
  Use sliders to change the Study Hours and Sleep Hours inputs, as well as the weights and biases of each neuron.

- **Dynamic Neuron Values:**  
  Watch the values inside each neuron update as you adjust the parameters. The color of the neurons also changes to reflect their output value.

- **Multiple Activation Functions:**  
  Select between Sigmoid, ReLU, and Tanh for the hidden layer to see how different non-linear functions affect the network's behavior.

- **Toggleable Connections:**  
  Show or hide the lines connecting the neurons. When visible, the lines are colored to indicate the sign of the weight (blue for positive, red for negative).

---

## Python Code (`lesson2.py`)

The Python script is a simplified, console-based version of the core MLP logic from the web app. It shows you exactly how the **forward pass calculation** works without the visual interface.

### Features

- **Pure Logic:**  
  Focuses on the mathematical operations behind the MLP.

- **Modular Functions:**  
  Contains clear functions for `sigmoid`, `relu`, and `tanh`, demonstrating how different activation functions are applied.

- **Easy to Modify:**  
  You can easily change the input values, weights, and biases directly in the code to experiment with the network's behavior.

---

## How to Run

### Web Visualization

1. Copy the code from the HTML file.  
2. Save it as `lesson2.html` on your computer.  
3. Open the file in any modern web browser (e.g., Chrome, Firefox, Safari).

### Python Code

1. Copy the code from the Python file.  
2. Save it as `lesson2.py`.  
3. You need the `numpy` library. If you don't have it, open your terminal or command prompt and run:

```bash
pip install numpy

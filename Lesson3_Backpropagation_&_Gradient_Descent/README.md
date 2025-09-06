# Lesson 2: Visualizing Backpropagation & Gradient Descent



This project is an **interactive web application** designed to visualize the core concepts of **backpropagation** and **gradient descent** in a simple neural network.  
It serves as an **educational tool** to make these abstract machine learning algorithms more intuitive and understandable.

---

## Features

### Neural Network Visualization
The main panel displays a **neural network with three layers**:  
- Input Layer  
- Hidden Layer  
- Output Layer  

**Components:**
- **Neurons**  
  - Circles represent neurons.  
  - Their color and internal value change to show activation.  
  - For the output neuron:  
    - Green → High probability of "Pass"  
    - Red → Low probability  

- **Weights**  
  - Lines connecting neurons are the **weights**.  
  - Thickness shows strength, color shows sign:  
    - Blue → Positive weight  
    - Red → Negative weight  

- **Biases**  
  - Each hidden/output neuron has a small **`b`**.  
  - This is a constant value that shifts the activation threshold.  

- **Forward & Backward Pass**  
  - **Forward Pass**: Input flows through the network to output.  
  - **Backward Pass** (Backpropagation): Error is propagated backward, gradients are calculated, and weights/biases are updated.  

---

### Gradient Descent Plot
This panel shows the **learning process** of the network.

- **Error Curve**  
  - Blue line = total error (loss).  
  - Should decrease over time as the network learns.  

- **White Dot**  
  - Shows the network’s **current position** on the curve.  
  - Moves step-by-step with gradient descent.  

- **Axes**  
  - Y-axis = Total Error  
  - X-axis = Training Epochs  

---

### Interactive Controls
The app provides several interactive options:

- **Data Sliders**  
  Adjust input values (e.g., *Study Hours*, *Sleep Hours*) and see predictions change live.

- **Pre-defined Data Points**  
  Select from example datasets to test network performance on different cases.

- **Activation Function**  
  Change the activation function (Sigmoid, Tanh, ReLU) and observe its effect.

- **Learning Rate**  
  Adjust the learning rate (\(\eta\)):  
  - Large → Faster but unstable learning  
  - Small → Slower but precise learning  

- **Training Buttons**  
  - Run step-by-step training  
  - Run auto-training loop across epochs  

---

## Core Concepts Explained

### 1. Forward Pass
Input values flow through the network to make a prediction.  
Each neuron computes:  

\[
z = \sum_{i=1}^{n} w_i x_i + b
\]  

Then applies an activation function \(f(z)\) → activation \(a\).  

---

### 2. Backpropagation
The prediction \(\hat{y}\) is compared with the actual target \(y\).  
The **error** is propagated backward:  

\[
\frac{\partial L}{\partial W^{(l)}} = 
\frac{\partial L}{\partial a^{(l)}} \cdot
\frac{\partial a^{(l)}}{\partial z^{(l)}} \cdot
\frac{\partial z^{(l)}}{\partial W^{(l)}}
\]  

Each weight learns how much it contributed to the error.  

---

### 3. Gradient Descent
Uses gradients to update parameters:  

\[
W := W - \eta \frac{\partial L}{\partial W}, \quad 
b := b - \eta \frac{\partial L}{\partial b}
\]  

The network takes steps downhill on the error curve until it finds the minimum.  

---

## How to Run
1. Open `index.html` in a web browser.  
2. No server or extra dependencies required.  
3. Start exploring forward pass, backpropagation, and gradient descent live.  

---

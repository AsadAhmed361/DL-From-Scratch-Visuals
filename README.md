# DL-From-Scratch-Visuals
I'm learning ML/DL by generating interactive visualizations with AI. This repo documents my journey, transforming abstract concepts like neurons and backpropagation into tangible, hands-on experiences. Each lesson pairs an interactive web demo with the foundational Python code, making complex ideas simple to see and understand.


# 🧠 ML/DL Visualized: A Learning Journey with AI

Welcome to **ML/DL Visualized**, a repository dedicated to my journey of learning Machine Learning and Deep Learning concepts with the help of AI-generated interactive visualizations.

My mission is to transform abstract and complex AI concepts into clear, visual, and hands-on experiences. I'm using AI to generate code for interactive web pages (HTML, CSS, and JavaScript) that allow me to manipulate parameters and see their effects in real time. This approach makes learning more intuitive and fun.

---

## **Lesson 1: The Single Neuron** 💡

This lesson explores the most fundamental building block of a neural network: a single neuron. It's the simplest way to understand how a model takes in data, processes it, and produces an output.

### **The Interactive Visualization**

The core of this lesson is a web-based visualizer that shows a single neuron with two inputs: **Study Hours** and **Sleep Hours**. You can adjust sliders to change these inputs, as well as the neuron's **weights** and **bias**. The visualization updates instantly, showing you how each parameter affects the final "Pass Probability."



By interacting with this tool, you can see these core concepts in action:

* **Inputs (x1, x2):** The data points we feed into the neuron.
* **Weights (w1, w2):** The importance of each input. Notice how a positive weight for "Study Hours" increases the output, while a negative weight for "Sleep Hours" decreases it.
* **Bias (b):** A value that adjusts the neuron's base output, independent of the inputs.
* **Activation Function:** The final step that squashes the output into a meaningful range (e.g., a probability between 0 and 1 using Sigmoid).

### **Code Breakdown**

This lesson includes two primary components:

1.  **`lesson1.html`:** The interactive visualization built with HTML, CSS, and JavaScript. This file contains all the front-end code that brings the neuron to life.
2.  **`lesson1_code.py`:** The equivalent Python code using the `numpy` library. This is the **computational backbone** of the neuron, showing the same logic in a scripting environment. It's a great way to connect the visual front-end with the back-end mathematical operations.

By having both the interactive visualizer and the raw Python code, I'm able to learn on multiple levels: seeing the concepts, manipulating them, and then understanding the underlying math.

### **How to Use**

Simply open `lesson1.html` in your web browser to start interacting with the single neuron. You can also run `lesson1_code.py` in your terminal or a Jupyter notebook to see the calculations performed step-by-step.

### **My Learning Journey**

I'm excited to continue this project and visualize more complex topics, such as:

* **Multi-Layer Perceptrons (MLPs)**: Visualizing a network with multiple neurons and hidden layers.
* **Cost Functions:** Seeing how a model's "error" is calculated.
* **Backpropagation:** Visualizing how a neural network learns by adjusting weights and biases based on the error.
* **Convolutional Neural Networks (CNNs):** Understanding how filters process images.

Stay tuned for more lessons!
<img width="827" height="614" alt="image" src="https://github.com/user-attachments/assets/510c66fe-ef00-4455-9f6b-21d2d31337fb" />

---

## **License**

This project is licensed under the MIT License.

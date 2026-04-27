# Handwritten Digit Recognition

A neural network built using TensorFlow/Keras that recognizes 
handwritten digits (0-9) with 92% accuracy on unseen test data.

## Overview
This project uses a fully connected neural network trained on the 
MNIST dataset — 60,000 handwritten digit images. The model learns 
to classify any handwritten digit from 0 to 9.

## Model Architecture
| Layer | Neurons | Activation |
|-------|---------|------------|
| Input | 784 | — |
| Hidden Layer 1 | 25 | ReLU |
| Hidden Layer 2 | 15 | ReLU |
| Output Layer | 10 | Softmax |

## Results
| Metric | Value |
|--------|-------|
| Test Accuracy | 92.18% |
| Test Loss | 0.302 |

## How it works
1. Each 28x28 image is flattened into a vector of 784 pixels
2. Pixel values are fed into the neural network
3. The output layer produces 10 probabilities (one per digit)
4. The digit with the highest probability is the prediction

## Dataset
MNIST — built into TensorFlow/Keras, loads automatically:
```python
tf.keras.datasets.mnist.load_data()
```
70,000 images total — 60,000 training, 10,000 testing.

## How to run
1. Install dependencies:
pip install tensorflow matplotlib numpy
2. Open and run `digit_recognition.ipynb` in Jupyter Notebook

## What I learned
- Building a neural network using TensorFlow/Keras
- Forward propagation and activation functions (ReLU, Softmax)
- Multiclass classification with categorical cross entropy
- Visualizing model predictions on real image data

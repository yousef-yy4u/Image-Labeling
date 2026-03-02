# Image Labeling Project

The goal of this project is to soldify my understanding of data processing and ML concepts.

I will be using the CIFAR-10 dataset to train and evaluate different Neural Network classifiers. I’ll be implementing functions that are needed to initialize, train, evaluate, and make predictions. Additionally I’ll be analyzing the results and explaining my observations.

## 1. Data Processing
First, the data is loaded and preprocessed (transformed to Tensors and Normalized). Then, it is separated into training and testing sets. Because using the full CIFAR-10 dataset takes too long, I broke down the dataset into smaller subsets:
* Training Set: 20,000 samples
* Validation Set: 10,000 samples
* Testing Set: 10,000 samples

This ensures that the model can be trained on smaller batches (batch size of 250) for faster iterations.

## 2. Model Architecture
I defined a `DynamicFeedforwardNN` that allows for a flexible number of hidden layers, customizable input/hidden sizes, and dropout probabilities for regularization. It uses the `SGD` optimizer and `CrossEntropyLoss`.

Two weight initialization methods were tested: zero weight initialization and random weight initialization.

## 3. Training and Validation (Sigmoid vs. ReLU)
I implemented a training function that tracks training and testing error rates per epoch.

### Baseline Model
Trained a baseline feedforward model (Input: 3072, Hidden: 100, Layers: 1, Activation: Sigmoid).
* **Training Error:** ~0.69
* **Testing Error:** ~0.68

### Activation Function Comparison
Switching from Sigmoid to **ReLU** activation resulted in a significant jump in learning efficiency. The Sigmoid model struggled with the vanishing gradient problem in deeper networks, eventually randomly guessing with ~0.90 error.
* **ReLU Performance:** Solved the vanishing gradient issue, dropping the training error to ~0.33 and improving the F-measure to ~0.66.
* **Overfitting:** However, ReLU introduced clear overfitting. Testing error plateaued near 0.55 while training error continued to decrease.

## 4. Hyperparameter Sweeps
* **Learning Rate Sweep:** Tested learning rates from 0.05 to 1.0. Higher learning rates initially decreased error, with the lowest error achieved at `lr=0.5` (~0.46 training, ~0.55 testing). However, higher learning rates also triggered overfitting faster.
* **Hidden Layers Sweep:** Tested models with 1 to 8 hidden layers. Models with 1 or 2 hidden layers performed significantly better. Deeper networks (3+ layers) suffered from high error rates (~0.83-0.90) due to increased complexity and optimization difficulty.

## 5. K-Fold Cross Validation
Applied K-Fold Cross Validation to optimize parameters. 
* Due to computational constraints, the grid search space was limited (10 epochs, 100 nodes).
* **Selected Configuration:** 100 Nodes, 0.1 Learning Rate, 10 Epochs.
* **Results:** Average Test Error of 0.7478 and Test F1 of 0.2144.
* **Observation:** Limiting epochs to 10 for grid search led to underfitting compared to the single-split ReLU trained for 30 epochs (Test Error: 0.5545). However, k-fold provides a more honest evaluation of the model's predictive power on unseen data.

## 6. Convolutional Neural Network (CNN)
As a final comparison, a Convolutional Neural Network (CNN) was trained.
* **Architecture:** 2 Convolutional Layers with Max Pooling, followed by 2 Fully Connected Layers.
* **Performance:** The CNN outperformed the feedforward model significantly, cutting down the error rate by ~26% (down to ~0.48) and doubling the F-measure (~0.51).
* **Efficiency:** Training was much faster (47 seconds) compared to the feedforward k-fold tuning (243 seconds), as CNNs reuse weights in filters and retain the 2D spatial layout of image pixels, unlike feedforward networks that flatten images and lose spatial context.

### Reference

Learning Multiple Layers of Features from Tiny Images(https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009.

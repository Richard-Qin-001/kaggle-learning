# Kaggle Digit Recognizer Solution

This repository contains the solution for the [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) competition. The proposed model achieves an impressive accuracy of **0.99728** on the public leaderboard without using external datasets or pseudo-labeling.

## 1. Data Processing & Augmentation

To prevent overfitting and enable the model to learn robust, invariant features of handwritten digits, we implemented a targeted data augmentation pipeline using `torchvision.transforms`:

*   **RandomAffine**: Random rotations (up to 15 degrees), translations (10% vertically and horizontally), scaling (90% to 110%), and shearing (10 degrees) mimic natural variations in handwriting styles.
*   **RandomPerspective**: Adds a slight perspective distortion (scale=0.2, p=0.5) to account for digits written at an angle.
*   **Test-Time Augmentation (TTA)**: During inference, we apply subtle affine transformations (slight shifts, scales, and rotations) to generate multiple views of the same test image. The final prediction is formulated by averaging the predicted probabilities across all views.

## 2. Network Design & Architecture

The solution utilizes a model ensemble consisting of two distinct, highly optimized Convolutional Neural Network (CNN) architectures to capture diverse feature representations: 

### Model A: ResNet-MNIST (with SE Blocks)
A lightweight Residual Network customized for 28x28 grayscale images. 
*   **Activation:** Swapped standard ReLU with `GELU` for smoother gradient flow.
*   **Attention Mechanism:** Integrated **Squeeze-and-Excitation (SE) Blocks** (with a reduction ratio of 4) into the residual connections. This enables the network to perform dynamic channel-wise feature recalibration, allowing it to focus on the most informative feature maps without choking the early layers.

### Model B: WideConvNet
A custom VGG-style network with wider convolutional layers. 
*   **Feature Extraction:** Four conceptual blocks of `Conv2d -> BatchNorm2d -> GELU` with increasing channel depths (64 -> 128 -> 256), followed by Maxpooling and Dropout.
*   **Classifier:** A robust fully-connected head with heavy dropout (`p=0.5`) to enforce ultimate feature generalization before the final unnormalized logits.

## 3. Training Strategy

*   **Ensemble learning via 5-Fold Cross-Validation:** We trained both model architectures over 5 stratified folds to ensure that predictions are generalized across the entire training distribution, resulting in an ensemble of 10 models (5 ResNets + 5 WideConvNets).
*   **Loss Function:** Used `CrossEntropyLoss` with `label_smoothing=0.1` to prevent overconfidence in predictions and combat overfitting.
*   **Optimizer & Scheduler:** Trained with `AdamW` (Weight Decay=1e-4) alongside a `CosineAnnealingLR` scheduler over 30 epochs per fold for stable and steady convergence.

# Deep Learning CheatSheet

The following summary is inspired by: [https://www.globalsqa.com/deep-learning-cheat-sheet/](Deep Learning CheatSheet)

## 1. Introduction

Deep Learning is a subset of machine learning that uses artificial neural networks (Current state of the art) to solve problems. The two main benefits are:

1. Automatic feature extraction: Unlike traditional machine learning techniques (Input -> Feature Extraction -> Algorithm -> Output), it removes the need of manual feature extraction/engineering.
2. Non-Linearity


## 2. Training Neural Networks

### 2.1 Main Training Algorithm

Each 

### 2.2 Regularization

### 2.3 Tips

### 2.4 Transfer learning

#### 2.4.1 Fine-tuning

#### 2.4.2 Feature extraction

### 2.5 Multi-task Learning


## 3. Activation Functions

## 4. Loss Functions

## 5. Optimizer algorithms

## 6. Evaluation Metrics

## 7. Architectures
 
### 7.1 Feed-Forward Neural Network (FNN or MLP)

**Multi-Layer Perceptrons (MLPs)** are the foundation of deep learning models. These networks are composed of multiple layers of interconnected perceptrons, where each perceptron is a computational unit that performs a weighted sum of the input, applies an activation function, and passes the result to the next layer.

An MLP consists of:

- **Input Layer**: Accepts the input features.
- **Hidden Layers**: Contain perceptrons that learn complex patterns and transformations. Usually named *fully-connected/Dense layers*
- **Output Layer**: Produces the final output, typically with an activation function like **Sigmoid** (for binary classification) or **Softmax** (for multi-class classification).

**Activation functions** like **ReLU**, **Sigmoid**, and **Tanh** are used to introduce **non-linearity** into the network, allowing MLPs to model more complex relationships compared to linear models. An MLP without activation functions can be proven to be equivalent to a composition of linear models, which are a linear model. This estimators do not need any special assumption of the data or the model like linear or generalized linear models:

$$
\hat{y} = 𝑓(w_1\cdot x_1 + w_2\cdot x_2 + ... + 𝑏) \implies \text{Dense Layer, f is the activation function}
$$

$$
\hat{y} = w_1\cdot x_1 + w_2\cdot x_2 + ... + 𝑏 \implies \text{Linear Model}
$$

#### Example: Simple MLP for Binary Classification

Consider an MLP with 2 input neurons, 2 hidden neurons (hidden layer), and 1 output neuron (final layer).

#### 1. **Input Vector**
$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.8 \end{bmatrix}
$$

#### 2. **Hidden Layer**
Weights and biases:
$$
W_1 = \begin{bmatrix} 0.4 & -0.6 \\ 0.7 & 0.1 \end{bmatrix}, \quad \mathbf{b}_1 = \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix}
$$

Hidden layer output:
$$
\mathbf{z}_1 = W_1 \cdot \mathbf{x} + \mathbf{b}_1 = \begin{bmatrix} -0.18 \\ 0.23 \end{bmatrix}
$$

Activation (ReLU):
$$
\mathbf{h} = \max(0, \mathbf{z}_1) = \begin{bmatrix} 0 \\ 0.23 \end{bmatrix}
$$

#### 3. **Output Layer**
Weights and bias:
$$
W_2 = \begin{bmatrix} 0.9 & -0.3 \end{bmatrix}, \quad b_3 = 0.05
$$

Output:
$$
z_2 = W_2 \cdot \mathbf{h} + b_3 = -0.019
$$

Activation (Sigmoid):
$$
\hat{y} = \sigma(z_2) = \frac{1}{1 + e^{0.019}} \approx 0.495
$$



### 7.2 Recurrent Neural Network (RNN) 

### 7.2.1 Recurrent Unit (RNN)

### 7.2.2 Gated Recurrent Unit (GRU)

### 7.2.3 Long-Short Term Memory Unit (LTSM) 


### 7.3 Convolutional networks

1. These networks make use of the **convolution** operation for images: combine two functions to produce a third function. The convolution is used to extract features from an input image **F** using a filter/kernel **K**. Each filter extracts a feature-map, which is the result of the convolution operation and represents a smaller-processed image. **(if the filter is 2x2 and the image 64x64, then the feature-map of that convolution size is (64-2+1) * (64-2+1) = 63x63)**

 For a single-channel image with just 1 filter: 
$$  Y(i,j) = Convolution = (F * K)(i,j) = \sum_m \sum_n  F(i+m, j+n) * K(m,n) + b$$


For colors with images (2 dimensions and color channel) and multiple filters, the learning process would approximate the filters weights and the bias terms associated with each filter.

$$
    Y(i,j,k) = Convolution = (F * K)(i,j, k) =  \sum_m \sum_n \sum_c F(i+m, j+n, c) * K(m,n,c,k) + b_k
$$

**This equation represents sliding a filter across the input image, multiplying element-wise, summing up, and adding bias.**

After performing convolution, an activation function layer (ReLU / tanh) is followed to introduce non-linearity in the feature-map (given that convolutions are linear operations):

$$ Z(i,j,k)=max(0,Y(i,j,k)) $$

The bigger the number of filters (32, 64, 128...) the more low level features we can capture in the images such as edges, textures, diagonals... and other visual patterns. If this convolutional layer contains 32 filters, then it has 32 * 3 (i)* 3 (j) * 3 (c) = 864 learnable parameters. In addition 32 bias terms , one for each filter.


2. **Stride** controls how much the filter moves in each step horizontally and vertically. A stride of 1 means the filter moves pixel by pixel, while a stride of 2 skips every other pixel, reducing spatial dimensions.

3. **Padding** Padding helps control the output size of the feature map. "Valid" padding applies no extra pixels, shrinking the output feature-map, while "Same" padding (zero-padding) preserves the input size by adding zeros around the border.

4. **Pooling** layers are used to reduce the dimensionality of feature-maps and speed-up computations after the activation function. Pooling can be done through Max, Min, Avg and global pooling that reduces the whole feature-map to 1 value.

5. **Flatten** layers are used to transform tensors to simple vectors that can be processed by fully connected layers.


The hyperparameters of convolutional layers are:

1. Number of filters
2. Stride
3. Activation function

#### 7.3.1 Convolutional Neural Network (CNN)

Convolutional Neural Networks (CNNs) are widely used in computer vision tasks, such as image classification. For example, ResNet and VGG architectures have been successfully applied to large-scale datasets like ImageNet, where they classify images into thousands of categories with high accuracy. These models extract hierarchical features—starting with edges and textures in early layers and progressing to complex objects in deeper layers—making them powerful for real-world applications like autonomous vehicles, medical diagnosis, and facial recognition. Therefore they are often mixed with other architechture such as MLP or RNN to cover such use-cases.

Example of a simple CNN architechture (High-level process below with Backpropagation) operations performed by hand (Forward and Backward propagations), processing a greyscale input image X for a binary classification task:

- 1️⃣ Convolution (2×2 only 1 filter) → Feature detection
- 2️⃣ ReLU Activation → Introduces non-linearity
- 3️⃣ Pooling (2×2 max pooling) → Reduces dimensions
- 4️⃣ Flatten → Converts feature maps into a unidimensional vector 
- 5️⃣ Fully connected layer → Classification and decision-making 
- 6️⃣ Softmax → Converts output into class probabilities
- 7️⃣ Backpropagation → Updates weights to minimize loss

Given this network N and image X, the result of $N(X) = \{0.31, 0.69\}$, therefore the image X belongs with higher probability to class 2. Below all the process is described:

$$
X =
\begin{bmatrix}
1 & 2 & 3 & 0 \\
4 & 5 & 6 & 1 \\
7 & 8 & 9 & 2 \\
0 & 1 & 2 & 3
\end{bmatrix}
$$
The X image is 4x4 and only 1 filter of 2x2 is used with bias, therefore padding isn't needed
$$
K =
\begin{bmatrix}
1 & 0 \\
-1 & 2
\end{bmatrix}
,  b = 0
$$

$$
Y_{\text{conv}}(i,j) =
\begin{bmatrix}
(1 \times 1) + (2 \times 0) + (4 \times -1) + (5 \times 2) + 0 & (2 \times 1) + (3 \times 0) + (5 \times -1) + (6 \times 2) + 0 & \dots \\
(4 \times 1) + (5 \times 0) + (7 \times -1) + (8 \times 2) + 0 & (5 \times 1) + (6 \times 0) + (8 \times -1) + (9 \times 2) + 0 & \dots \\
\dots & \dots & \dots
\end{bmatrix}
$$

Applying the convolution operation yields the following feature map


$$
Y_{\text{conv}}(i,j) =
\begin{bmatrix}
7 & 9 & 8 \\
10 & 12 & 11 \\
5 & 6 & 7
\end{bmatrix}
$$

Now an activation function such as ReLU is applied to introduce non-linearity to the convolution.
$$
Z_(i, j) = \max(0, Y_{\text{conv}}(i, j))
$$

$$
Z(i,j) =
\begin{bmatrix}
7 & 9 & 8 \\
10 & 12 & 11 \\
5 & 6 & 7
\end{bmatrix}
$$
To reduce dimensionality of the feature-map, max-pooling 2x2 is applied:
$$
V_{\text{pool}}(i,j) =
\begin{bmatrix}
\max(7, 9, 10, 12) & \max(9, 8, 12, 11) \\
\max(10, 12, 5, 6) & \max(12, 11, 6, 7)
\end{bmatrix}
$$

$$
V_{\text{pool}} =
\begin{bmatrix}
12 & 12 \\
12 & 12
\end{bmatrix}
$$
Before using the fully connected MLP layer, we need to perform vectorization of the matrix $W_{pool}$, Flattening
$$
V_{Flattened} = [12, 12, 12, 12]
$$

Finally,the output feature-map of the CNN is ingested by the fully connected-layer (2 perceptrons) which performs a regular linear feed-forward operation that outputs 2 log-odds numbers. This is the pre-activation function $Z = W \cdot X + b$

$$
\begin{bmatrix}
w_1 & w_2 & w_3 & w_4 \\
w_5 & w_6 & w_7 & w_8
\end{bmatrix}
\cdot
\begin{bmatrix}
12 \\
12 \\
12 \\
12
\end{bmatrix}
+
\begin{bmatrix}
b_1 \\
b_2
\end{bmatrix}
=
\begin{bmatrix}
z_1 \\
z_2
\end{bmatrix}
$$

$$
z_1 = (0.1 \times 12) + (0.2 \times 12) + (-0.1 \times 12) + (0.05 \times 12) + 0.5 = 3.5
$$

$$
z_2 = (-0.3 \times 12) + (0.2 \times 12) + (0.4 \times 12) + (0.1 \times 12) - 0.5 = 4.3
$$

After the fully-connected layer, the activation function **softmax** is used to change the scale of the log-odds to probabilities

$$
P_1 = \frac{e^{3.5}}{e^{3.5} + e^{4.3}} = \frac{33.1}{33.1 + 73.4} = 0.31
$$

$$
P_2 = \frac{e^{4.3}}{e^{3.5} + e^{4.3}} = \frac{73.4}{33.1 + 73.4} = 0.69
$$

After performing forward propagation, loss must be computed and weight update if we are actively training the network and not just infering:
$$
W_{\text{new}} = W_{\text{old}} - \eta \frac{\partial \text{Loss}}{\partial W}
$$

#### 7.3.2 Capsule Networks (CapsNets)

Capsule Networks (CapsNets) are an advanced neural network architecture designed to address limitations in traditional Convolutional Neural Networks (CNNs), particularly in handling spatial hierarchies and viewpoint variations. Unlike CNNs, which rely on max-pooling for dimensionality reduction, CapsNets use **capsules**, which encode both **features** and their **spatial relationships**.


CapsNets replace the standard CNN layers with a **capsule-based structure**, which consists of:

1. **Convolutional Layer:** Extracts low-level features, similar to CNNs.
2. **Primary Capsules Layer:** Groups convolutional features into small vector capsules.
3. **Higher-Level Capsules (DigitCaps):** Captures spatial relationships between features using **dynamic routing**.
4. **Reconstruction Network:** Uses a **decoder** to reconstruct the input image, aiding in interpretability.

---

#### Step-by-Step Architecture of CapsNet

##### **1. Input Layer**
- Takes in a **grayscale image** (e.g., MNIST 28×28).
- The image is **normalized** before being passed into the network.

##### **2. Convolutional Layer**
- A **standard convolutional layer** extracts **basic features** like edges and textures.
- Uses **ReLU activation** to introduce non-linearity.
- Example:
  - **256 filters** of **9×9** kernel applied with **stride 1**.

**Conv Output Shape:** `(20 × 20 × 256)`

---

##### **3. Primary Capsules Layer**
- Instead of scalars, features are now grouped into **capsules** (small vector groups).
- Each capsule represents a **part of an object** and encodes **both presence & orientation**.
- Implemented using **convolutional capsules** (each acting as a mini-network).
- Example:
  - **32 capsule groups**
  - **Each capsule = 8D vector**
  - **Kernel size: 9×9, Stride: 2**
  - Output: **(6 × 6 × 32 capsules) → Reshaped to (1152 capsules × 8D)**

**Capsule Output Shape:** `(1152, 8)`

---

##### **4. Digit Capsules (Higher-Level Capsules)**
- Each **higher-level capsule** represents **a whole object**, such as a digit in MNIST.
- Uses **dynamic routing** (instead of max-pooling) to determine **which lower capsules send their output**.
- Example:
  - **10 capsules (one per digit)**
  - Each capsule outputs a **16D vector**
  - **Routing Algorithm:** **Squash Function** ensures outputs are **unit-length vectors**

**DigitCaps Output Shape:** `(10, 16)`

---

##### **5. Capsule Routing Mechanism**
- Unlike CNNs, where signals flow in a fixed way, **CapsNets use routing-by-agreement**:
  1. **Low-level capsules predict higher-level capsules' output.**
  2. **Capsules compare agreement (dot product similarity).**
  3. **High-agreement capsules strengthen their connection.**
  4. **Squash activation ensures output is within unit-length.**

$v_j = \frac{||s_j||^2}{1 + ||s_j||^2} \cdot \frac{s_j}{||s_j||}$


---

##### **6. Decoder (Reconstruction Network)**
- To **enhance learning**, CapsNets include a **decoder** that reconstructs the original input.
- The decoder is a **small MLP with three fully connected layers**:
  1. **First layer**: 512 neurons
  2. **Second layer**: 1024 neurons
  3. **Output layer**: Reshaped into an **image (28×28 pixels for MNIST)**

**Decoder Output:** `(28, 28)`

---

| **Layer**            | **Type**  | **Details**  |
|----------------------|----------|-------------|
| **Input**           | Image    | `28×28` (MNIST) |
| **Conv Layer**      | CNN      | `9×9` kernel, 256 filters |
| **Primary Capsules** | Capsules | 32 capsules, 8D vectors |
| **Digit Capsules**  | Capsules | 10 capsules, 16D vectors |
| **Decoder**        | MLP      | Reconstructs the image |

---



### 7.4 Generative Networks

#### 7.4.1 Auto Encoders (AE)

#### 7.4.2 Variational Auto Encoders (VAE)

#### 7.4.3 Generative Adversarial Networks (GAN)

#### 7.4.4 Difusion Model 

#### 7.4.5 Deep Belief Networks



### 7.5 Transformers

#### 7.5.1 Attention

#### 7.5.2 Transformer


### 7.6 Graph Models

#### 7.6.1 Graph Neural Networks (GNN)

#### 7.6.2 Graph Convolutional Networks (GCN)

#### 7.6.3 Graph Attention Networks (GAT)




### 7.7 Hybrid Models 

#### 7.7.1 Neural Architecture Search (NAS)

#### 7.7.2 Memory augmented neural networks (MANN)

#### 7.7.3 Neural Ordinary Differential Equations (ODE)

#### 7.7.4 Multi-modal Models



### 7.8 Reinforcement Models

#### 7.8.1 Deep-Q Networks (DQN)

#### 7.8.2 Actor Critic 


### 7.9 Self-Organizing Maps

### 7.10 Self Supervised Learning


 
## 8. Other important Deep Learning Concepts

### 8.1 Few-Shot and Zero-Shot

### 8.2 Mixture of Experts (MoE)

### 8.3 Retrieval Augmented Generation (RAG)



## 9. Explainability and Interpretability

### 9.1 SHAP 

### 9.2 LIME

### 9.3 Attention VIZ




## 10 Deep Learning Hardware & Compilatin

### 10.1 Hardware

### 10.2 Compilation

### 10.3 Compression




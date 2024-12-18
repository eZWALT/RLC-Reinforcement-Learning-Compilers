# Resources 

In this directory you can find the resources that I've been using to learn optimization algorithms and theory:

## Introduction to Deep Learning (15h)
Check the **deep-learning/** directory 

## Reinforcement Learning (150h)
- [Reinforcement Learning - Wikipedia](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [Reinforcement Learning Specialization - Coursera](https://www.coursera.org/specializations/reinforcement-learning)
- [Deep Reinforcement Learning Course - Hugging Face](https://huggingface.co/learn/deep-rl-course/unit0/introduction)
- [Reinforcement Learning - DeepMind / UCL](https://www.davidsilver.uk/teaching/)

The courses take up to 90 + 25 + 40 hours to complete at a chill pace (Avoid binging education :). Notes can be found at the **notes** directory.


## Evolutionary algorithms (15h)

- [Evolutionary Algorithms - Optimization lectures playlist](https://www.youtube.com/watch?v=P2nLBTJdx_A&list=PLJEWP9Z0q94AXyg8ZTi-V0xwFMgbVt5wZ)
- [Evolutionary Strategies Guide - LilLog](https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/)
- [Evolutionary Strategies Tutorial - Machine learning mastery](https://machinelearningmastery.com/evolution-strategies-from-scratch-in-python/)

The whole optimization lectures take approximately 10 hours to watch  and the guides + tutorial would be 4h 
The evolution algorithms hierarchy looks like this:

```
Evolutionary Algorithms
├── Genetic Algorithms (GA)
│   ├── Binary Encoding
│   ├── Real-Valued Encoding
│   └── Hybrid Variants
├── Evolution Strategies (ES)
│   ├── (μ + λ) Selection
│   ├── (μ, λ) Selection
│   └── Self-Adaptive Variants
├── Genetic Programming (GP)
│   └── Tree-Based Representations
└── Differential Evolution (DE)
    └── Vector-Based Mutation

```



### Extra

The optimization landscape looks something like this (Partially incomplete)
```
Artificial Intelligence (AI)
├── Symbolic AI (Good Old-Fashioned AI - GOFAI)
│   └── Rule-Based Systems
│       ├── Expert Systems
│       └── Knowledge Representation
│           ├── Ontologies
│           └── Logic Programming
│
├── Machine Learning (ML)
│   ├── Supervised Learning
│   │   ├── Regression
│   │   │   ├── Linear Regression
│   │   │   └── Logistic Regression
│   │   ├── Classification
│   │   │   ├── Decision Trees
│   │   │   ├── Support Vector Machines (SVM)
│   │   │   ├── k-Nearest Neighbors (k-NN)
│   │   │   ├── Random Forests
│   │   │   └── Gradient Boosting (e.g., XGBoost, CatBoost)
│   │   └── Neural Networks (NN)
│   │       ├── Shallow Neural Networks
│   │       └── Deep Learning (DL)
│   │           ├── Convolutional Neural Networks (CNNs)
│   │           ├── Recurrent Neural Networks (RNNs)
│   │           │   ├── Long Short-Term Memory (LSTM)
│   │           │   └── Gated Recurrent Units (GRU)
│   │           ├── Transformers
│   │           │   ├── BERT, GPT, etc.
│   │           └── Generative Models
│   │               ├── Variational Autoencoders (VAE)
│   │               └── Generative Adversarial Networks (GANs)
│   │
│   ├── Unsupervised Learning
│   │   ├── Clustering
│   │   │   ├── k-Means
│   │   │   ├── Hierarchical Clustering
│   │   │   └── DBSCAN
│   │   ├── Dimensionality Reduction
│   │   │   ├── Principal Component Analysis (PCA)
│   │   │   └── t-SNE, UMAP
│   │   └── Anomaly Detection
│   │       ├── Isolation Forests
│   │       └── Autoencoders
│   │
│   ├── Semi-Supervised Learning
│   │   └── Pseudo-Labeling, Self-Training
│   │
│   └── Reinforcement Learning (RL)
│       ├── Model-Free RL
│       │   ├── Value-Based
│       │   │   ├── Q-Learning
│       │   │   └── Deep Q-Networks (DQN)
│       │   ├── Policy-Based
│       │   │   ├── REINFORCE
│       │   │   └── Actor-Critic (A2C, A3C)
│       │   └── Advanced Algorithms
│       │       ├── Proximal Policy Optimization (PPO)
│       │       └── Soft Actor-Critic (SAC)
│       ├── Model-Based RL
│       │   └── Planning (e.g., Dyna-Q, Monte Carlo Tree Search)
│       └── Hierarchical RL
│
├── Optimization Algorithms
│   ├── Gradient-Based Optimization
│   │   ├── Gradient Descent (GD)
│   │   ├── Stochastic Gradient Descent (SGD)
│   │   ├── ADAM...
│   ├── Gradient-Free Optimization
│   │   ├── Evolutionary Algorithms (EA)
│   │   │   ├── Genetic Algorithms (GA)
│   │   │   ├── Evolution Strategies (ES)
│   │   │   └── Differential Evolution (DE)
│   │   ├── Swarm Intelligence
│   │   │   ├── Particle Swarm Optimization (PSO)
│   │   │   └── Ant Colony Optimization (ACO)
│   │   ├── Bayesian Optimization
│   ├── Heuristic Search
│   │   ├── Greedy Search
│   │   ├── A* Search
│   │   ├── Best-First Search
│   │   ├── Simulated Annealing
│   │   ├── Tabu Search
│   │   ├── Beam Search

```
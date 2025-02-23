# RESuM: Rare Event Surrogate Model

A Machine Learning-Based Surrogate Model for Rare Event Design Optimization  

## Overview
The Rare Event Surrogate Model (RESuM) is designed to optimize detector designs and simulation-driven systems where the objective function involves rare event probabilities.  

Rare Event Design (RED) Problems  
Many physics and engineering problems involve optimizing designs to minimize rare but critical events, such as background contamination in particle detectors.  
However, evaluating these events through simulations is often computationally expensive, leading to high variance and low signal-to-noise ratio.  

## What RESuM Does  
RESuM provides a machine learning-based surrogate model that significantly reduces computational costs while maintaining accuracy. It integrates:
- Pre-trained Conditional Neural Processes (CNPs) for incorporating prior knowledge.
- Multi-Fidelity Gaussian Process (MF-GP) modeling to blend low- and high-fidelity simulations efficiently.
- Adaptive sampling strategies to guide expensive simulations only where needed.

Reference Paper:  
[RESuM: A Rare Event Surrogate Model](https://openreview.net/pdf?id=lqTILjL6lP)  
If you use this code, please cite our paper!

## Features
- Optimizes rare event-driven design problems  
- Reduces computational cost while preserving accuracy  
- Combines physics priors with deep learning models  
- Multi-fidelity approach for integrating different levels of simulations  
- Scalable to other domains in rare event search and optimization  

## Installation

For local development, install the `resum` library:
```bash
pip install -e .
```

Then run the Jupyter notebooks in the [examples folder](https://github.com/annkasch/resum/tree/main/examples)!

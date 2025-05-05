# Dimensional Mean Squared Error (DMSE)



**DMSE: A Geometrically-Informed Scalar Loss Function for Multi-Dimensional Regression**

This repository contains the official implementation of **Dimensional Mean Squared Error (DMSE)**, a novel scalar loss function that captures both the magnitude and direction of multi-dimensional targets in regression tasks. It enables modeling multi-dimensional outputs using a single neuron while preserving geometric relationships between target vectors.


## 🔍 Overview

Traditional loss functions like Mean Squared Error (MSE) treat each output dimension independently. DMSE overcomes this limitation by incorporating **vector-based comparisons** into loss computation, aiming to align both magnitude and direction of predictions with multi-dimensional targets.

The DMSE loss is formally defined as:

$$
\mathcal{L}_{DMSE} = \left( \mathbf{y}_{true}^\top \mathbf{y}_{pred} - \mathbf{y}_{true}^\top \mathbf{y}_{true} \right)^2
$$

An alternative formulation used in implementation is:

$$
\mathcal{L}_{DMSE} = \left( (\mathbf{y}_{pred} \cdot \mathbf{y}_{true}) + (\text{sign}(\mathbf{y}_{pred}) \cdot \text{sign}(\mathbf{y}_{true})) - (\mathbf{y}_{true}^2 + \text{sign}(\mathbf{y}_{true})^2) \right)^2
$$

Where:
- $\mathbf{y}_{true}$ is the ground truth vector  
- $\mathbf{y}_{pred}$ is the predicted vector  
- $	ext{sign}(\cdot)$ represents the element-wise sign function

These formulations focus on both the **geometric alignment** and **sign consistency** between predicted and actual vectors.


## 🧪 Usage & Demo

The full implementation and evaluation of DMSE is contained in the following Jupyter notebook:

```bash
Main.ipynb
```

## 🛠️ Implementation Details

A Long Short-Term Memory (LSTM) neural network was implemented to evaluate the proposed DMSE loss function in a time series forecasting task. The model predicts the price change of Bitcoin using historical price and volume data.

The architecture consists of:
- Multiple LSTM layers followed by dense layers
- A two-dimensional output: percentage change and direction of movement

The model was trained under two configurations:
- Loss: Mean Squared Error (MSE) vs. Dimensional Mean Squared Error (DMSE)
- Optimizers: Stochastic Gradient Descent (SGD) and Adam

Training was executed using a Jupyter notebook:  
📄 `Main.ipynb`

## 📊 Experimental Results

We compared the DMSE and MSE loss functions across two scenarios using the same LSTM model:

1. **SGD Optimizer**
2. **Adam Optimizer**

For each, the model was trained with MSE and then retrained with DMSE.

Key observations:
- **Volatility sensitivity**: DMSE produced outputs with greater variance, better reflecting price dynamics.
- **Directional accuracy**: The alignment between predicted and target directional indicators improved using DMSE.

Visualizations (found in `images/` directory) show:
- Comparison of prediction patterns
- Enhanced geometric matching using DMSE

These results support the claim that DMSE improves both the sensitivity to directional changes and the fidelity of multi-dimensional predictions.

### 📷 Visual Results

![MSE with Adam Optimizer](https://github.com/mahan100/DMSE/blob/essay/actual_dloss_mse_adam.png?raw=true)
*Figure: Prediction using MSE loss with Adam optimizer*

![MSE vs DMSE with SGD Optimizer](https://github.com/mahan100/DMSE/blob/essay/actual_dloss_mse_sgd.png?raw=true)
*Figure: MSE prediction using SGD optimizer*

![DMSE vs MSE with SGD Optimizer](https://github.com/mahan100/DMSE/blob/essay/actual_dmse_mse_sgd.png?raw=true)
*Figure: DMSE prediction comparison using SGD optimizer*


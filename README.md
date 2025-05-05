# Dimensional Mean Squared Error (DMSE)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**DMSE: A Geometrically-Informed Scalar Loss Function for Multi-Dimensional Regression**

This repository contains the official implementation of **Dimensional Mean Squared Error (DMSE)**, a novel scalar loss function that captures both the magnitude and direction of multi-dimensional targets in regression tasks. It enables modeling multi-dimensional outputs using a single neuron while preserving geometric relationships between target vectors.

## 🔍 Overview

Traditional loss functions like Mean Squared Error (MSE) treat each output dimension independently. DMSE overcomes this limitation by incorporating **vector-based comparisons** into loss computation, measuring:

- **Length Difference** (Magnitude Mismatch)
- **Angle Difference** (Directional Misalignment)

The result is a loss function that improves alignment with multi-dimensional target dynamics.

## 📈 Use Case

We demonstrate DMSE in a **Bitcoin price prediction** task using LSTM, where the model predicts:

- Price change percentage
- Direction of change

Our experiments show that DMSE outperforms traditional MSE in sensitivity and accuracy with respect to target volatility and directional behavior.

## 📁 Repository Structure

```
.
├── dmse_loss.py        # DMSE loss function implementation
├── model.py            # LSTM model for regression
├── train.py            # Training script
├── data/               # Sample or processed dataset
├── utils/              # Utility scripts
└── README.md           # Project documentation
```

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/mahan100/DMSE.git
cd DMSE
pip install -r requirements.txt
```

### Training the Model

```bash
python train.py --epochs 100 --loss dmse
```

### Evaluating the Model

```bash
python evaluate.py --model_path ./checkpoints/model.pth
```

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@misc{mohseni2025dmse,
  title={Dimensional Mean Squared Error (DMSE): A Geometrically-Informed Scalar Loss Function for Multi-Dimensional Regression},
  author={Mehdi Mohseni Mahani},
  year={2025},
  howpublished={\url{https://github.com/mahan100/DMSE}}
}
```

## 📬 Contact

For questions or feedback, please contact:  
**Mehdi Mohseni Mahani**  
📧 mohsenimehdi1367@gmail.com

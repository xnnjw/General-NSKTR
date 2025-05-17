# NS-KTR: Non-negative Structured Kruskal Tensor Regression

## Setup and Requirements
- MATLAB
- Tensor Toolbox (v2.6 or v3.6)
- External methods (included in repository):
  - TensorReg-master
  - tensor_toolbox-2.6
  - AOAS21-QuantileTR

## Quick Start

1. Clone the repository
2. Run the demos:

### Demo: Synthetic Data
```matlab
% Tests NS-KTR on synthetic signals (Gradient, Floor, Wave, Fading Cross)
% Compares LS, EN, nEN, FL, and nFL regularization approaches
run Demo1_Synthetic.m
```

Results will be displayed as plots and tables comparing the performance of different methods.

## Citation
If you use this code, please cite [PAPER CITATION].

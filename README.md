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

### Demo 1: Synthetic Data
```matlab
% Tests NS-KTR on synthetic signals (Gradient, Floor, Wave, Fading Cross)
% Compares LS, EN, nEN, FL, and nFL regularization approaches
run Demo1_Synthetic.m
```

### Demo 2A: HSI Regression
```matlab
% Applies NS-KTR to HSI data for regression tasks
% Parameters to modify:
% - sampling_rate: 25%, 50%, or 75%
% - target_index: 1-4 (GrainWeight, Gsw, PhiPS2, Fertilizer)
% - rank: tensor rank (default=1)
run Demo2A_HSI_linear.m
```

### Demo 2B: HSI Classification
```matlab
% Applies NS-KTR to HSI data for classification tasks
% Parameters to modify:
% - sampling_rate: 25%, 50%, or 75%
% - target_index: 5-8 (Heerup, Kvium, Rembrandt, Sheriff)
% - rank: tensor rank (default=1)
run Demo2B_HSI_logistic.m
```

Results will be displayed as plots and tables comparing the performance of different methods.

## Citation
If you use this code, please cite [PAPER CITATION].

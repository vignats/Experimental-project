# Experimental Project: Blood Pressure Estimation from Pulse Arrival Time (PAT)

This project aims to estimate blood pressure (BP) from pulse arrival time (PAT) data using various machine learning models.

## Overview

The main script includes functions for model training, evaluation, and visualization, utilizing the `Preprocess` and `Model` classes from the `models.py` module. Here's an overview of the key classes and their functionalities:

### Classes (from `models.py`):

1. **`Preprocess`:**
    - Contains methods to preprocess the PAT data, including interpolation and outlier removal.

2. **`Model`:**
    - Initializes and trains machine learning models for BP estimation, handling data splitting, normalization, and accuracy computations.

### Functions:

1. **`plot_predicted_values(y_pred, y_test, model)`:**
    - Visualizes the predicted values against actual blood pressure values.
    
2. **`compute_accuracies(model)`:**
    - Computes RMSE and R2 accuracies based on different preprocessing configurations.
    - Saves the results in an Excel file named `accuracy.xlsx`.

3. **`k_fold()`:**
    - Performs k-fold cross-validation to assess model performance using different fold sizes.

## Usage:

1. Update the `path` variable in the `Model` class with the path to your PAT data CSV file.
2. modify the `model` variable to select the desired machine learning model. 
3. Modify the parameters in the `Model` class instantiation to configure preprocessing options and the machine learning model.
4. Run the script to execute the functions and generate accuracy results or visualization.

## Inputs and Outputs

### Inputs:
- **CSV Data:** Contains PAT and BP information for model training and evaluation.

### Outputs:
- **`accuracy.xlsx`:** Excel file storing RMSE and R2 results based on different preprocessing and split configurations.

Refer to the `models.py` module for detailed implementations of the `Preprocess` and `Model` classes.

# LSTM modeling

### Modelling on one dataset
The script `Final-code-LSTM-with-preprocess.py`use the `Preprocess` to do the preprocessing of the data.

### Modelling on supplementary datasets 
The script `Autoregressive-LSTM-with-TensorFlow-for-other-datasets.py` isn't linked to other classes.
It allow to test the model on new datasets.

## Outputs of the two scripts
- Plot of comparaison between predicted BP and true BP.
- Visualization of training set, testing set and predict one. 
- RMSE evolution between test and train sets with `summarize_diagnostics`.


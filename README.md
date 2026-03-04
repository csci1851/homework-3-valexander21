[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/FuXjn4oO)
# Homework 3: Support Vector Machines (SVMs)

## Overview
In this homework, you will explore Support Vector Machines (SVMs), a classical machine learning method. 
You will implement and evaluate SVMs for both classification and regression tasks using real biomedical datasets. 
You will:

- Load and inspect biomedical datasets.
- Perform classification (heart disease).
- Perform regression (biological aging prediction).
- Compare linear, polynomial, and RBF kernels.
- Visualize decision boundaries and regression fits.

---

## Datasets

You will work with two datasets:
1. Heart Disease Dataset (Classification)
- **Goal**: Predict presence/absence of heart disease.
- **Features**: Demographics, cholesterol, ECG results, etc.

2. Biological Aging Dataset (Regression)
- **Goal**: Predict biological age using molecular and physiological features.
- **Source**: High-dimensional gene expression dataset (GSE139307).
- **Features**: Several hundred genomic and biological markers, already preprocessed and normalized.

---

## Installation

Install dependencies using pip:

1. **Clone** this repo (first time only):
   ```bash
   git clone git@github.com:brown-csci1851/stencil.git
   cd stencil/homework3
   ```
   If you already cloned it, update and move into the homework folder:
   ```bash
   cd stencil
   git pull
   cd homework3
   ```
2. Create virtual environment:
    ```bash
    python -m venv .hw3
    ```
3. Install dependencies:
    ```bash
    source .hw3/bin/activate (Linux/MacOS) or .\.hw3\Scripts\activate
    pip install -r requirements.txt
    ```

After creating and activating the virtual environment, select it as the Jupyter kernel in `src/playground.ipynb` to run the notebook using the same installed dependencies.

---

## Tasks

You will complete the following and include them in your reflection:

- [ ] Load both datasets using `HW3DataLoader`.
- [ ] Inspect the datasets (shapes, missing values, feature distributions, class balance).
- [ ] Build **leakage-free Pipelines** that include: imputation → scaling → (optional PCA) → SVM/SVR.
- [ ] Explain SVM intuition: **margin vs. slack (C)**, **kernels (linear / polynomial / RBF)**, and **why scaling matters**.
- [ ] Train and tune **SVM classifiers** (heart disease) with:
  - [ ] Linear kernel
  - [ ] Polynomial kernel (tune degree)
  - [ ] RBF kernel (tune γ)
- [ ] Evaluate classification using metrics such as:
  - [ ] Accuracy, F1
  - [ ] ROC-AUC or PR-AUC
  - [ ] Confusion matrix + ROC/PR curve
- [ ] Handle missing values in the aging dataset using **column-wise mean imputation** (do not drop rows/cols).
- [ ] Train and tune **SVR regressors** (biological aging) with linear / polynomial / RBF kernels.
- [ ] Evaluate regression using:
  - [ ] MAE, RMSE, $R^2$
  - [ ] Parity plot (predicted vs actual) + residual plot
- [ ] Show hyperparameter effects with visuals (examples: performance vs. C, γ, polynomial degree).
- [ ] Compare kernels and summarize which worked best for each task.

## Final Reflection

You will then write a **2–3 page PDF reflection** that includes **figures** and **interpretation** of your results. Your write-up should clearly reference the plots, tables, and metrics you generated (not just final numbers). 

- **Data & preprocessing:** what features you used, missingness, scaling, PCA choice (if used).
- **Modeling & hyperparameters:** which kernel/params worked best for classification vs regression, and why.
- **Metrics & visuals:** include key scores + confusion matrix/ROC/PR (classification) and parity/residuals (regression).
- **Sensitivity:** how results changed with scaling/PCA and with C/γ/degree.
---

## Expected Skills

By the end of this homework, you should be able to:

* Use SVMs for both classification and regression.
* Understand kernel functions and their role in SVMs.
* Visualize and interpret decision boundaries and predictions.

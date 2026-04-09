# Credit Card Default Prediction & Interpretability Framework:

This repository contains the implementation of a high-precision predictive framework designed to identify credit card default risks.
The project focuses on balancing predictive power through non-linear modeling with a custom-built interpretability layer for financial transparency.

## Project Overview
Financial institutions face significant challenges in managing credit exposure. This project addresses those risks by:

+ Predictive Modeling: Utilizing a Support Vector Machine (SVM) with a Radial Basis Function (RBF) kernel to identify potential defaults among 30,000 credit card clients.
+ Granular Interpretability: Implementing a custom-built SHAP (SHapley Additive exPlanations) framework from scratch to provide local explanations for "black-box" model outputs.

## Key FeaturesHigh-Precision Classification:

Engineered an RBF-SVM model to capture complex, non-linear relationships within historical financial data, achieving an overall accuracy of 82%.
Custom Explainability Engine: Developed a manual implementation of the Shapley value formula to decompose individual predictions into feature-specific contributions.
Risk Driver Identification: Successfully mapped the top influential features—such as payment history ($X6$) and credit limit ($X1$)—to individual risk scores, enabling stakeholders to understand the "why" behind every prediction.
Memory-Efficient Computation: Designed a 2D/3D slicing pattern for coalition storage to overcome environment memory limitations when calculating exact Shapley values for a subset of features.

## Dataset & Preprocessing

The model utilizes the UCI Credit Card Default dataset, featuring 30,000 instances and 23 features.
Categorical Encoding: Implemented one-hot encoding for features like education, gender, and repayment status.
Feature Scaling: Applied MinMaxScaler to normalize numerical data (e.g., age, bill amounts) for optimal SVM performance.
Class Imbalance Handling: Utilized stratified sampling to maintain class proportions across a 70/30 train-test split.

## Technical StackLanguage: 
PythonLibraries: Scikit-learn (SVM modeling), Pandas/NumPy (Data manipulation), Matplotlib/Seaborn (Feature correlation & visualization).
Core Algorithm: Support Vector Machine with RBF Kernel.
Explainability: Custom SHAP Implementation based on the Shapley weight formula:

$$\phi_{i}=\sum_{S\subseteq N\backslash\{i\}}\frac{|S|!(N-|S|-1)!}{N!}[f(S\cup\{i\})-f(S)]$$

## Project Results

The framework provides detailed local interpretability for individual cases:
Example 1 (Non-Default): High age ($X5 = 56$) was identified as a primary driver for a positive prediction, correlating with higher financial responsibility.
Example 2 (Default): Recent payment delays (specifically $X6$, payment due for 2 months) were correctly identified as the heaviest contributors lifting the risk score to a "Default" status.

## How to Use
Open the provided .ipynb file in Google Colab or a local Jupyter environment.
Ensure dependencies (sklearn, pandas, numpy) are installed.
Run the preprocessing cells to clean and scale the UCI dataset.
Execute the SVM training cell to generate the predictive model.
Use the SHAP implementation section to generate local explanations for specific test instances.




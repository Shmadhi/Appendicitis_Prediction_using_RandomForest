# Pediatric Appendicitis Prediction using Random Forest
R-based data science project to build a predictive model for diagnosing pediatric appendicitis. Using clinical and lab data, I developed a Random Forest classifier to distinguish between appendicitis and no appendicitis cases. The project includes data cleaning, preprocessing, model tuning with cross-validation, and performance evaluation.

Pediatric Appendicitis Prediction using Random Forest
This repository contains an R-based data science project to build a predictive model for diagnosing pediatric appendicitis. Using clinical and lab data, we develop a Random Forest classifier to distinguish between appendicitis and no appendicitis cases. The project includes data cleaning, preprocessing, model tuning with cross-validation, and performance evaluation.

# Project Context
Appendicitis is a common cause of acute abdominal pain in children. Accurate and timely diagnosis is critical to avoid complications and unnecessary surgeries. This project applies machine learning techniques to pediatric patient data, aiming to support clinical decision-making by predicting appendicitis diagnosis based on clinical indicators such as appendix diameter, CRP levels, white blood cell counts, and others.

# Data
The dataset used (cleaned_patients.csv) includes pediatric patient records with relevant clinical and laboratory measurements.

Missing values are handled via mean (numeric) and mode (categorical) imputation.

Column names are cleaned and standardized for ease of processing.

# Key Steps
1. Data Cleaning & Preprocessing

Load and clean column names.

Convert relevant columns to numeric, fixing formatting issues.

Impute missing values for numeric and categorical features.

Prepare features and target variable (diagnosis).

Impute missing values for numeric and categorical features.

Prepare features and target variable (diagnosis).

2. Data Splitting:

Split data into 80% training and 20% testing sets, stratified by diagnosis.

3. Model Setup:

Create a preprocessing recipe to normalize numeric variables and convert categorical predictors to dummy variables.

Define a Random Forest classifier with tuning parameters (trees, min_n, mtry).

Combine recipe and model into a workflow.

4. Hyperparameter Tuning:

Use 5-fold cross-validation to tune hyperparameters over a regular grid.

Optimize based on ROC AUC.

5. Final Model & Evaluation:

Train final model on full training set with best parameters.

Evaluate on test set using metrics including accuracy, ROC AUC, F1-score, and specificity.

Display confusion matrix heatmap and ROC curve with AUC annotation.

Optionally show variable importance plot if the vip package is installed.

# Visualization
Confusion matrix heatmap visualizes true vs predicted diagnoses.

ROC curve illustrates the trade-off between sensitivity and specificity, with AUC shown on the plot.

Variable importance plot highlights the most influential predictors in the model.

# Usage
Place cleaned_patients.csv in your working directory.

Run the R script or R Markdown notebook sequentially to reproduce the analysis and model training.

Ensure all listed R packages are installed (tidymodels, dplyr, janitor, ranger, ggplot2, etc.).

# Requirements
R (version >= 4.0 recommended)

R packages: tidymodels, dplyr, janitor, ranger, ggplot2, vip (optional)

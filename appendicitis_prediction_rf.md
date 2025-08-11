```{r}
# Load necessary libraries
library(tidymodels)       # For modeling, workflows, and %>% operator
library(dplyr)            # For data manipulation
library(readr)            # For reading CSV files
library(janitor)          # For cleaning column names
library(ranger)           # For random forest engine
library(ggplot2)          # For visualization
```

```{r}
# Step 1: Load the data and clean column names
CLEANED_PATIENTS <- read.csv("cleaned_patients.csv")

data <- CLEANED_PATIENTS %>% 
  clean_names()  # Clean the column names (convert to lowercase with underscores)

# Step 2: Convert relevant columns to numeric (fix for the "new levels" warning)
data <- data %>%
  mutate(
    length_of_stay = as.numeric(gsub("[^0-9.]", "", length_of_stay)),
    crp = as.numeric(gsub("[^0-9.]", "", crp)),
    thrombocyte_count = as.numeric(gsub("[^0-9.]", "", thrombocyte_count)),
    weight = as.numeric(gsub("[^0-9.]", "", weight)),
    rbc_count = as.numeric(gsub("[^0-9.]", "", rbc_count)),
    rdw = as.numeric(gsub("[^0-9.]", "", rdw)),
    bmi = as.numeric(gsub("[^0-9.]", "", bmi)),
    hemoglobin = as.numeric(gsub("[^0-9.]", "", hemoglobin))
  )

# Step 3: Impute missing values for the specified columns
# Numeric columns to impute
impute_numeric <- c("length_of_stay", "crp", "thrombocyte_count", 
                    "weight", "rbc_count", "rdw", "bmi", "hemoglobin")

# Categorical columns to impute
impute_categorical <- c("psoas_sign")

# Impute numeric columns (mean imputation)
data <- data %>%
  mutate(across(all_of(impute_numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Impute categorical columns (mode imputation)
mode_impute <- function(x) {
  if (is.factor(x) || is.character(x)) {
    mode_val <- names(sort(table(x), decreasing = TRUE))[1]
    return(ifelse(is.na(x), mode_val, x))
  }
  return(x)
}

data <- data %>%
  mutate(across(all_of(impute_categorical), mode_impute))
```

```{r}
# Step 4: Prepare the dataset for training
# Define the target variable and predictor variables
target_variable <- "diagnosis"
predictor_variables <- c(
  "appendix_diameter", "crp", "wbc_count", 
  "thrombocyte_count", "peritonitis", "age", "weight", 
  "severity", "migratory_pain"
)

# Convert Diagnosis to a factor and ensure "appendicitis" is the first level
data <- data %>%
  mutate(diagnosis = factor(diagnosis, levels = c("appendicitis", "no appendicitis")))

# Select only the target and predictor variables
data <- data %>%
  select(all_of(c(target_variable, predictor_variables)))
```

```{r}
# Step 5: Split the data into training and testing sets
set.seed(123)  # For reproducibility
data_split <- initial_split(data, prop = 0.8, strata = diagnosis)
train_data <- training(data_split)
test_data  <- testing(data_split)

# Step 6: Create a recipe for preprocessing
data_recipe <- recipe(diagnosis ~ ., data = train_data) %>%
  step_normalize(all_numeric_predictors()) %>%  # Normalize numeric predictors
  step_dummy(all_nominal_predictors(), -all_outcomes())

# Step 7: Specify the random forest model
rf_model <- rand_forest(
  trees = tune(),                
  min_n = tune(),                
  mtry = tune()                  
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

# Step 8: Create a workflow to combine the recipe and model
rf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(data_recipe)
```

```{r}
# Step 9: Set up cross-validation
set.seed(123)
cv_folds <- vfold_cv(train_data, v = 5, strata = diagnosis) 

# Step 10: Set up a grid of hyperparameters to search
rf_grid <- grid_regular(
  trees(range = c(100, 1000)),       
  min_n(range = c(5, 25)),           
  mtry(range = c(3, length(predictor_variables))), 
  levels = 5                        
)

# Step 11: Tune the model
set.seed(123)
tune_results <- tune_grid(
  rf_workflow,
  resamples = cv_folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc, accuracy) 
)
```

```{r}
# Step 12: Select the best hyperparameters
best_params <- select_best(tune_results, metric = "roc_auc")

# Finalize the workflow with the best parameters
final_rf_workflow <- finalize_workflow(rf_workflow, best_params)

# Step 13: Fit the final model to the training data
final_rf_model <- fit(final_rf_workflow, data = train_data)

# Step 14: Evaluate on the test set
final_predictions <- predict(final_rf_model, test_data, type = "prob") %>%
  bind_cols(predict(final_rf_model, test_data)) %>%
  bind_cols(test_data) 

# Evaluate model performance with additional metrics
final_metrics <- final_predictions %>%
  metrics(truth = diagnosis, estimate = .pred_class, .pred_appendicitis) %>%
  bind_rows(
    final_predictions %>% 
      yardstick::f_meas(truth = diagnosis, estimate = .pred_class) %>%
      mutate(.metric = "f1_score")
  )

# Print the performance metrics
print(final_metrics)
```

```{r}
# Calculate confusion matrix
confusion_mat <- final_predictions %>%
  conf_mat(truth = diagnosis, estimate = .pred_class)

# Calculate specificity from the confusion matrix
specificity <- confusion_mat %>% 
  summary() %>%
  filter(.metric == "specificity") %>%
  select(.estimate) %>%
  pull()  # Extracts the value instead of a tibble

# Display the confusion matrix
confusion_mat %>%
  autoplot(type = "heatmap")

# ROC Curve: Visualizes the trade-off between sensitivity and specificity.
# Calculate ROC data
roc_data <- final_predictions %>%
  roc_curve(truth = diagnosis, .pred_appendicitis)

# Calculate AUC
auc_val <- final_predictions %>%
  roc_auc(truth = diagnosis, .pred_appendicitis) %>%
  pull(.estimate) %>%
  round(3)

# Plot
autoplot(roc_data) +
  labs(
    title = "ROC Curve",
    x = "False Positive Rate",
    y = "True Positive Rate",
    color = "Diagnosis"
  ) +
  annotate("text", x = 0.75, y = 0.25, label = paste("AUC =", auc_val), size = 5, color = "blue") +
  theme_minimal()

# Optional: Show variable importance plot
if ("vip" %in% rownames(installed.packages())) {
  library(vip)
  final_rf_model %>% 
    extract_fit_parsnip() %>%
    vip(num_features = 10)  # Top 10 most important variables
}
```

# Logistic Regression Model Analysis Extensions

This repository contains extensions to the `LogisticRegressionModel` class that add comprehensive analysis capabilities for model understanding and optimization.

## Overview

The following new capabilities have been added:

1. **Feature Importance Analysis** - Understand which features contribute most to predictions
2. **Ablation Studies** - Measure the impact of removing features or feature groups
3. **Sampling Strategy Comparison** - Find the optimal approach for handling class imbalance
4. **Sensitivity Analysis** - Explore the impact of different hyperparameters
5. **Model Retraining** - Apply insights to create an optimized model

## Integration Instructions

### Step 1: Add the new methods to `LogisticRegressionModel`

Add the methods from `Enhanced LogisticRegressionModel Class` to your existing `LogisticRegressionModel` class. 

These methods are designed to seamlessly integrate with the existing class structure and include:
- `get_feature_importance()`
- `perform_ablation_study()`
- `perform_individual_feature_ablation()`
- `compare_sampling_strategies()`
- `perform_sensitivity_analysis()`
- `retrain_with_recommendations()`

### Step 2: Required Imports

Ensure the following imports are added to your model file:

```python
from collections import defaultdict
from sklearn.model_selection import cross_validate
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
```

### Step 3: Run the Jupyter Notebook

The `logistic_regression_experiments.ipynb` notebook demonstrates how to use these new methods to:

1. Analyze a trained logistic regression model
2. Generate insights about feature importance and model behavior
3. Create data-driven recommendations for model improvements
4. Implement those recommendations by retraining an optimized model

## Using the Notebook

The notebook is structured to guide you through the complete model analysis process:

1. **Setup** - Load data and train a base model
2. **Feature Importance** - Identify which features drive predictions
3. **Ablation Studies** - Test the impact of removing features
4. **Sampling Strategies** - Compare approaches for handling class imbalance
5. **Sensitivity Analysis** - Explore hyperparameter impacts
6. **Recommendations** - Create data-driven recommendations
7. **Retraining** - Apply recommendations to create an optimized model
8. **Evaluation** - Compare original and optimized model performance

## Requirements

- scikit-learn >= 0.24.0
- pandas >= 1.2.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- imbalanced-learn >= 0.8.0
- joblib >= 1.0.0

## Customization

The notebook is designed to be adaptable to your specific dataset and use case:

- Modify the feature lists to match your data structure
- Adjust the hyperparameter ranges to explore relevant values
- Change the evaluation metrics to align with your business objectives
- Update the visualization settings to emphasize key findings

## Next Steps

After completing this analysis, consider:

1. Validating the optimized model on additional datasets
2. Comparing the logistic regression model with other algorithms
3. Developing a monitoring strategy for the deployed model
4. Creating documentation for stakeholders explaining the model's behavior
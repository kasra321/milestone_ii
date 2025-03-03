# Logistic Regression Evaluation Results

## Metrics

| Metric                    | Score    |
|---------------------------|----------|
| Validation Accuracy       | 0.7300 |
| Validation F1 Score (weighted) | 0.7332 |
| Validation Precision (weighted) | 0.7482 |
| Validation Recall (class 1) | 0.7692 |
| Binary Precision (class 1) | 0.6250 |
| Binary F1 Score (class 1) | 0.6897 |
| ROC-AUC Score             | 0.8217 |

## Detailed Classification Report

```
              precision    recall  f1-score   support

           0       0.83      0.70      0.76       122
           1       0.62      0.77      0.69        78

    accuracy                           0.73       200
   macro avg       0.73      0.74      0.73       200
weighted avg       0.75      0.73      0.73       200

```

## Sample of Predictions in Original Labels

`[False False  True  True  True] ...`

# XGBoost Evaluation Results

## Metrics

| Metric                    | Score    |
|---------------------------|----------|
| Validation Accuracy       | 0.6900 |
| Validation F1 Score (weighted) | 0.6774 |
| Validation Precision (weighted) | 0.6828 |
| Validation Recall (class 1) | 0.8361 |
| Binary Precision (class 1) | 0.7083 |
| Binary F1 Score (class 1) | 0.7669 |
| ROC-AUC Score             | 0.7295 |

## Detailed Classification Report

```
              precision    recall  f1-score   support

           0       0.64      0.46      0.54        78
           1       0.71      0.84      0.77       122

    accuracy                           0.69       200
   macro avg       0.68      0.65      0.65       200
weighted avg       0.68      0.69      0.68       200

```

## Sample of Predictions in Original Labels

`['HOME' 'HOME' 'HOME' 'HOME' 'HOME'] ...`

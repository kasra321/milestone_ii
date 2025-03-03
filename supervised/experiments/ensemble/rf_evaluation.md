# Random Forest Evaluation Results

## Metrics

| Metric                    | Score    |
|---------------------------|----------|
| Validation Accuracy       | 0.6100 |
| Validation F1 Score (weighted) | 0.5795 |
| Validation Precision (weighted) | 0.5860 |
| Validation Recall (class 1) | 0.8197 |
| Binary Precision (class 1) | 0.6410 |
| Binary F1 Score (class 1) | 0.7194 |
| ROC-AUC Score             | 0.6359 |

## Detailed Classification Report

```
              precision    recall  f1-score   support

           0       0.50      0.28      0.36        78
           1       0.64      0.82      0.72       122

    accuracy                           0.61       200
   macro avg       0.57      0.55      0.54       200
weighted avg       0.59      0.61      0.58       200

```

## Sample of Predictions in Original Labels

`['ADMITTED' 'HOME' 'ADMITTED' 'HOME' 'HOME'] ...`

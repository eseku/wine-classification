# Model Evaluation

Discussion of how well the models perform on out-of-sample-data.

## Evaluation Procedure

Train test Split.

### Five models were used in total with 7 classification algorithms:

- Random Forest Classifier
- Gradient Boosting Classifier
- Decision Tree Classifier
- Ada Boost Classifier
- Xgboost Classifier
- Logistic Regression Classifier
- KNeighbors Classifier with K=5

### These 7 models were trained with the train set portion of the dataset.

Running main.py would print the accuracy score function. This is also known as accuracy training.

### Scores

The random forest classifier had the highest score with an accuracy score of 0.9225.

The XGBoost classifier had the second highest score of 0.9175

Next was the decision tree classifier with a score of 0.8975

Gradient boost Classifier and KNeighbors Classifier with a score of 0.8925.
The K value of the Kneighbors Classifier was increased to minimize overfitting,
that is to prevent the model from learning the noise of the data instead of the
signal of the data.

Logistic Regression Classifier and Ada Boost Classifier both with a score of 0.89.

## Conclusion

In Conclusion, the random forest classifier, with the most accuracy score on
out-of-sample data would be the most performant on external data prediction.

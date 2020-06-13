'''
10671406
CSCD 312 INTRODUCTION TO ARTIFICIAL INTELLIGENCE
END OF SEMESTER EXAMINATION QUESTION 1A
'''

import joblib
from trainmodels import X_test, y_test
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


# An empty uninitialized dictionary of models. 
models = {}

# A list of filenames of previously trained and saved models using joblib.
filenames = [
    'random_forest_classifier_model.sav',
    'xgboost_model.sav',
    'ada_boost_classifier_model.sav',
    'gradient_boost_classifier_model.sav',
    'decision_tree_classifier_model.sav',
    'logistic_regression_classifier.sav',
    'kneighbors_classifier.sav'
    ]

#block to populate dictionary with models saved in files.
for file in filenames:
    models[file[:-4]] = joblib.load(f"models/{file}")

scores = []
x = []

# printing the accuracy score for each model on the test data
for model in models:
    prediction = models[model].predict(X_test)
    print(model,":")
    score = accuracy_score(y_test, prediction)
    print(score)
    scores.append(score)
    x.append(model)




# Bar Chart to illustrate Model eval and performance
# using matplotlib
x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, scores, color="green")
plt.xlabel("Classifiers")
plt.ylabel("Classifier Scores")
plt.title("Model Eval")
plt.xticks(x_pos, x)
plt.show()

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb

#load csv file with pandas
df = pd.read_csv("winequality-red.csv")


'''
 To change task into a classification task, a new variable is created
 calld Pass. 
 If quality score of wine is greater than 7, then pass is awarded a 1(True)
 value for pass, which means it passes as good wine.
 '''
df['pass'] = [1 if x >= 7 else 0 for x in df['quality']]


# Separating feature variable matrix X and target variable Y
# where y is the new variable created called Pass
X = df.drop(['quality','pass'], axis = 1)
y = df['pass']

# print(df['goodquality'].value_counts())

# Normalizing feature variables
X_features = X
X = StandardScaler().fit_transform(X)

'''
 Splitting the data into test and train portions, giving test a size of 25%
 This is for model evaluation purposes. To see how well the model would perform
 on data that was not part of the training data.
 The test size argument specifies what portion of data you want to split for testing
 purposes
 Random state argument specifies in what way your data would be split.
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)


'''
 Model training and saving
 All 7 models used are trained on the training set and then saved into an external file
'''
xgboost_model = xgb.XGBClassifier(random_state=1)
xgboost_model.fit(X_train, y_train)
joblib.dump(xgboost_model, "models/xgboost_model.sav")

decision_tree_classifier_model = DecisionTreeClassifier(random_state=1)
decision_tree_classifier_model.fit(X_train, y_train)
joblib.dump(decision_tree_classifier_model, "models/decision_tree_classifier_model.sav")

random_forest_classifier_model = RandomForestClassifier(random_state=1)
random_forest_classifier_model.fit(X_train, y_train)
joblib.dump(random_forest_classifier_model, "models/random_forest_classifier_model.sav")

gradient_boost_classifier_model = GradientBoostingClassifier(random_state=1)
gradient_boost_classifier_model.fit(X_train, y_train)
joblib.dump(gradient_boost_classifier_model, "models/gradient_boost_classifier_model.sav")

ada_boost_classifier = AdaBoostClassifier(random_state=1)
ada_boost_classifier.fit(X_train, y_train)
joblib.dump(ada_boost_classifier, "models/ada_boost_classifier_model.sav")

logistic_regression_classifier = LogisticRegression(random_state=1)
logistic_regression_classifier.fit(X_train, y_train)
joblib.dump(ada_boost_classifier, "models/logistic_regression_classifier.sav")

kneighbors_classifier = KNeighborsClassifier(n_neighbors=12)
kneighbors_classifier.fit(X_train, y_train)
joblib.dump(kneighbors_classifier, "models/kneighbors_classifier.sav")

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 18:03:28 2022

@authors: Bryson Carroll, Enrique Guzman Barron
@email: jbc2@email.sc.edu, guzmanbe@email.sc.edu

Purpose: We have created a decision tree along with a graph of the feature importances that
cause lead our program to decide whether or not a death event is going to take place for a patient.
We have also created a random forest classifier and we output a grapch of the ROC curve to show
the accuracy of our predictions.  
"""

import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report
import math
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    f1_score,
)
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score
import os


# import data
file_path = os.getcwd() + '\heart_failure_clinical_records_dataset.csv'
df = pd.read_csv(file_path)

# shuffle rows
df = df.sample(frac=1)

# use first 1500 rows as training set
train_df = df.iloc[0:240]

# create test set
test_df = df.iloc[240:]

train_X = train_df[train_df.columns[:-1]]
train_Y = train_df["DEATH_EVENT"]
test_X = test_df[test_df.columns[:-1]]
test_Y = test_df["DEATH_EVENT"]

clf = tree.DecisionTreeClassifier()
clf.fit(train_X, train_Y)
plt.figure(figsize=(30, 8))
tree.plot_tree(clf, fontsize=5)

test_predict_Y = clf.predict(test_X)

print("Predictions: ", test_predict_Y)
print("Confusion matrix: ", confusion_matrix(test_Y, test_predict_Y, labels=[1,0]))


tp, fn, fp, tn = confusion_matrix(test_Y,test_predict_Y,labels=[1,0]).reshape(-1)
print('True Positives: ',tp)
print('False Negatives: ',fn)
print('False Positives: ',fp)
print('True Negatives: ',tn)

print('Classification Report: \n', classification_report(test_Y,test_predict_Y,labels=[1,0]))


# calculate via sklearn
precision = precision_score(test_Y, test_predict_Y, average=None)

# print feature importances plot
plt.figure()
plt.style.use("ggplot")
data = clf.feature_importances_
x_pos = [i for i, _ in enumerate(data)]
plt.bar(x_pos, data, color="green")
plt.xlabel("Variable")
plt.title("Feature importances")
plt.xticks(x_pos)
plt.show()


print('\nRandom Forest__________________________________')

# random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100)
pipe = make_pipeline(rf_classifier)
pipe.fit(train_X, train_Y)
pred_Y = pipe.predict(test_X)
print(f"Predicted pred_Y: {pred_Y}")

# ROC AUC score + curve
train_probs = pipe.predict_proba(train_X)[:, 1]
probs = pipe.predict_proba(test_X)[:, 1]
train_predictions = pipe.predict(train_X)
print(f"Train ROC AUC Score: {roc_auc_score(train_Y, train_probs)}")
print(f"Test ROC AUC Score: {roc_auc_score(test_Y, probs)}")
metrics.plot_roc_curve(pipe, test_X, test_Y)


plt.show()
